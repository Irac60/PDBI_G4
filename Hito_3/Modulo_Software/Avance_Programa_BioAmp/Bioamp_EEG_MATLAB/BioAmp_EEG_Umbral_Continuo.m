function BioAmp_EEG_Umbral_Continuo
% BIOAMP EEG: MONITOR CONTINUO + MARCADOR EXPLORATORIO DE AMPLITUD.
% Arduino SIN CAMBIOS: BioAmp_EEG_2_0.ino
% Formato: SEQ,RAW_mV,EEG_mV,Alfa_mV,AlfaRMS_mV,RIEL
%
% IMPORTANTE: ESTE PROGRAMA NO DETECTA NI DESCARTA CRISIS EPILEPTICAS.
% La marca de amplitud elevada NO es una alarma medica.
% U=79.1815 mV RMS se obtuvo de parpadeos/actividad sin etiqueta,
% no de crisis clinicamente confirmadas; NO es un umbral ictal.
% Fuente del umbral: BioAmp_EEG_Continuo_20261111_233742.csv.
% P97.5 de ventanas EEG 0.5-29.5 Hz, 128 muestras, salto 64,
% centros de ventana entre 14 y 40 s, SIN ETIQUETA de reposo.
%
% Seguridad: para electrodos humanos se requiere verificacion profesional
% de seguridad del montaje COMPLETO, incluyendo USB y PC si se emplean.
% Sin esa aprobacion usar solamente registros CSV historicos/equipo validado.
%
% USO: cierra Monitor Serie de Arduino, ejecuta este archivo y cierra la
% ventana de graficas para detener la adquisicion y cerrar el CSV.

puerto = "COM3";
Fs = 256;
baud = 230400;
ventanaSeg = 12;
umbralEEG_mV = 79.181464; % SOLO actividad EEG de amplitud elevada
N_rms = round(0.50*Fs);   % 128 muestras = 0.5 s
pasoRMS = round(0.25*Fs); % 64 muestras = 0.25 s, igual que analisis CSV
cap = ceil(ventanaSeg*Fs) + 64;

fprintf('MONITOR EEG EXPERIMENTAL | NO detector de crisis.\n');
fprintf('Umbral de amplitud EEG: %.2f mV RMS (no ictal).\n',umbralEEG_mV);
fprintf('Puertos disponibles:\n');
disp(serialportlist("available"));
fprintf('Cierra el Monitor Serie de Arduino antes de ejecutar.\n');

% Solo 0.5-29.5 Hz / 8-13 Hz YA FILTRADOS EN EL ARDUINO.
% No se aplica otro filtro: se evita cambiar las respuestas verificadas.
try
    s = serialport(puerto,baud,"Timeout",2);
    configureTerminator(s,"LF");
    flush(s);
catch ME
    error('No se pudo abrir %s. Cierra Monitor Serie y verifica COM: %s',puerto,ME.message);
end

nombreCSV = fullfile(pwd, ['BioAmp_EEG_Umbral_' datestr(now,'yyyymmdd_HHMMSS') '.csv']);
fid = fopen(nombreCSV,'w');
if fid < 0
    clear s;
    error('No se pudo abrir CSV para escritura: %s',nombreCSV);
end
fprintf(fid, ['SEQ,tiempo_s,raw_mV,eeg_mV,alfa_mV,alfaRMS_mV,' ...
    'RMS_EEG_0p5s_mV,ventana_nueva,amplitud_elevada,' ...
    'ventana_evaluable,cercaRiel,perdidas_acumuladas\n']);

% Cuatro graficas y colores siguiendo el estilo del monitor EMG.
f = figure('Name','BioAmp | EEG continuo y amplitud exploratoria', ...
    'Color',[0.055 0.075 0.12], 'NumberTitle','off', ...
    'Position',[75 50 1180 920]);
t = tiledlayout(f,4,1,'TileSpacing','compact','Padding','compact');
t.Title.String = 'BIOAMP EEG | Esperando muestras';
t.Title.FontSize = 14;
t.Title.Color = [0.95 0.96 1.00];

ax1=nexttile(t);
hRaw=plot(ax1,nan,nan,'Color',[0.95 0.83 0.49]);
ylabel(ax1,'RAW (mV)');
title(ax1,'ADS A0 sin filtro: salida del BioAmp');

ax2=nexttile(t);
hEEG=plot(ax2,nan,nan,'Color',[0.20 0.85 0.90]);
ylabel(ax2,'EEG (mV)');
title(ax2,'Senal filtrada 0.5-29.5 Hz (salida del BioAmp)');

ax3=nexttile(t);
hAlfa=plot(ax3,nan,nan,'Color',[0.80 0.55 1.00]);
ylabel(ax3,'Alfa (mV)');
title(ax3,'Componente 8-13 Hz');

ax4=nexttile(t);
hRMSEEG=plot(ax4,nan,nan,'Color',[1.00 0.73 0.30], 'LineWidth',1.8);
hold(ax4,'on');
hRMSAlfa=plot(ax4,nan,nan,'Color',[0.80 0.55 1.00], 'LineWidth',1.1);
hUmbral=plot(ax4,[0 ventanaSeg],[umbralEEG_mV umbralEEG_mV], '--', ...
    'Color',[0.95 0.45 0.45], 'LineWidth',1.3);
hold(ax4,'off');
ylabel(ax4,'RMS (mV)'); xlabel(ax4,'Tiempo de adquisicion (s)');
title(ax4,'RMS EEG 0.5 s / RMS alfa 1 s / umbral experimental');
legend(ax4,[hRMSEEG hRMSAlfa hUmbral], ...
    {'RMS EEG 0.5 s','RMS alfa 1 s','U=79.18 (NO crisis)'}, ...
    'Location','northwest','TextColor',[0.91 0.93 0.98], ...
    'Color',[0.085 0.11 0.18], 'EdgeColor',[0.40 0.48 0.60]);

axs=[ax1 ax2 ax3 ax4];
for k=1:4
    a=axs(k);
    a.Color=[0.085 0.11 0.18];
    a.XColor=[0.65 0.71 0.82];
    a.YColor=[0.65 0.71 0.82];
    a.GridColor=[0.40 0.48 0.60];
    a.GridAlpha=0.22;
    a.Title.Color=[0.91 0.93 0.98];
    a.FontSize=10;
    grid(a,'on');
    xlim(a,[0 ventanaSeg]);
end

% Buffer de visualizacion limitado a los ultimos 12 s.
tiempoV=nan(cap,1); rawV=nan(cap,1); eegV=nan(cap,1);
alfaV=nan(cap,1); alfaRMSV=nan(cap,1); rmsEEGV=nan(cap,1);

% Memoria circular para RMS y calidad de ventana: no crece con el tiempo.
potencias=zeros(N_rms,1);
malas=zeros(N_rms,1);
sumaPotencias=0;
sumaMalas=0;
ultimoRMS=nan;
ultimaEvaluable=false;
ultimaElevada=false;
n=0; primerSeq=nan; ultimoSeq=nan;
perdidas=0; totalRiel=0; fallos=0; ventanasElevadas=0;

fprintf('Adquisicion continua. Cierra la figura para finalizar.\n');
fprintf('Archivo CSV: %s\n',nombreCSV);

try
    while isgraphics(f)
        try
            linea=strtrim(char(readline(s)));
            fallos=0;
        catch ME
            if ~isgraphics(f), break; end
            fallos=fallos+1;
            fprintf('Lectura interrumpida (%d/5): %s\n',fallos,ME.message);
            if fallos>=5
                warning('Puerto serie: cinco errores seguidos.');
                break;
            end
            drawnow limitrate;
            continue;
        end

        v=sscanf(linea,'%f,%f,%f,%f,%f,%f');
        if numel(v)~=6 || any(~isfinite(v)) || v(1)<0 || v(1)~=floor(v(1))
            continue; % Incluye cabecera emitida por Arduino.
        end
        if ~isnan(ultimoSeq) && v(1)<=ultimoSeq
            warning('SEQ retrocedio: posible reinicio del ESP32.');
            break;
        end
        salto=0;
        if ~isnan(ultimoSeq)
            salto=max(0,v(1)-ultimoSeq-1);
            perdidas=perdidas+salto;
        end
        ultimoSeq=v(1);
        if isnan(primerSeq), primerSeq=v(1); end
        tm=(v(1)-primerSeq)/Fs;
        raw=v(2); eeg=v(3); alfa=v(4); alfaRMS=v(5);
        riel=double(v(6)~=0);
        totalRiel=totalRiel+riel;

        % Cada 128 muestras, RMS EEG sobre 0.5 s; cada 64 se evalua.
        % Ventana NO evaluable si contiene dato cerca de riel o salto SEQ.
        n=n+1;
        p=mod(n-1,N_rms)+1;
        sumaPotencias=sumaPotencias-potencias(p)+eeg^2;
        potencias(p)=eeg^2;
        sumaMalas=sumaMalas-malas(p)+double(riel~=0 || salto~=0);
        malas(p)=double(riel~=0 || salto~=0);
        ventanaNueva=0;
        if n>=N_rms && mod(n-N_rms,pasoRMS)==0
            ventanaNueva=1;
            ultimaEvaluable=(sumaMalas==0);
            if ultimaEvaluable
                ultimoRMS=sqrt(max(0,sumaPotencias)/N_rms);
                ultimaElevada=ultimoRMS>umbralEEG_mV;
                ventanasElevadas=ventanasElevadas+double(ultimaElevada);
            else
                ultimoRMS=nan;
                ultimaElevada=false;
            end
        end
        % El marcador NO indica crisis; es solo una bandera de investigacion.
        indicador=double(ultimaElevada && ultimaEvaluable && sumaMalas==0);
        calidad=double(ultimaEvaluable && sumaMalas==0);

        % Guardado continuo: todas las muestras sin crecer en memoria.
        fprintf(fid,'%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d,%d,%d\n', ...
            round(v(1)),tm,raw,eeg,alfa,alfaRMS,ultimoRMS, ...
            ventanaNueva,indicador,calidad,riel,round(perdidas));

        q=mod(n-1,cap)+1;
        tiempoV(q)=tm; rawV(q)=raw; eegV(q)=eeg;
        alfaV(q)=alfa; alfaRMSV(q)=alfaRMS; rmsEEGV(q)=ultimoRMS;

        if mod(n,26)==0 && isgraphics(f)
            idx=find(isfinite(tiempoV));
            [ts,orden]=sort(tiempoV(idx));
            ix=idx(orden);
            rv=rawV(ix); ev=eegV(ix); av=alfaV(ix);
            ar=alfaRMSV(ix); er=rmsEEGV(ix);
            set(hRaw,'XData',ts,'YData',rv);
            set(hEEG,'XData',ts,'YData',ev);
            set(hAlfa,'XData',ts,'YData',av);
            set(hRMSEEG,'XData',ts,'YData',er);
            set(hRMSAlfa,'XData',ts,'YData',ar);
            ini=max(0,tm-ventanaSeg);
            for k=1:4, xlim(axs(k),[ini ini+ventanaSeg]); end
            set(hUmbral,'XData',[ini ini+ventanaSeg]);
            margen=max(10,0.12*(max(rv)-min(rv)));
            ylim(ax1,[min(rv)-margen, max(rv)+margen]);
            escalaEEG=max(0.25,1.2*max(abs(ev)));
            escalaAlfa=max(0.10,1.2*max(abs(av)));
            ylim(ax2,[-escalaEEG escalaEEG]);
            ylim(ax3,[-escalaAlfa escalaAlfa]);
            vals=[ar(:);er(isfinite(er));umbralEEG_mV];
            ylim(ax4,[0 max(1,1.25*max(vals))]);

            % Estimacion 60 Hz de los ultimos 2 s de RAW (BioAmp OUT).
            sel60=ts>=tm-2;
            amp60=nan;
            if nnz(sel60)>=Fs
                x60=rv(sel60)-mean(rv(sel60));
                amp60=2*abs(sum(x60.*exp(-1i*2*pi*60*ts(sel60))))/nnz(sel60);
            end

            if n<N_rms
                estado='PREPARANDO RMS';
            elseif sumaMalas>0 || ~ultimaEvaluable
                estado='SENAL NO EVALUABLE';
            elseif ultimaElevada
                estado='AMPLITUD ELEVADA (NO CRISIS)';
            else
                estado='SIN EXCESO DE AMPLITUD';
            end
            t.Title.String=sprintf(['EEG | %s | %.1f s | RMS %.1f / U %.2f mV' ...
                ' | 60 Hz %.1f mV pico | perdidas %d | cerca de riel %d'], ...
                estado,tm,ultimoRMS,umbralEEG_mV,amp60,perdidas,totalRiel);
            drawnow limitrate;
        end
    end
catch ME
    warning('Captura interrumpida: %s',ME.message);
end

% Cierre de recursos aunque haya error durante el registro.
fclose(fid);
clear s;
if isnan(primerSeq), duracion=0; else, duracion=(ultimoSeq-primerSeq)/Fs; end
fprintf('\nEEG terminado | %d muestras | %.2f s | perdidas %d | riel %d.\n', ...
    n,duracion,perdidas,totalRiel);
fprintf('Ventanas de amplitud elevada: %d. NO son crisis.\n',ventanasElevadas);
fprintf('CSV guardado en:\n%s\n',nombreCSV);
end
