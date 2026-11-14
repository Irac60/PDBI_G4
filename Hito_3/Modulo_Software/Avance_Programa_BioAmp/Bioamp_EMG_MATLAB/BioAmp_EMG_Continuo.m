function BioAmp_EMG_Continuo
% BIOAMP EMG CONTINUO - BioAmp EXG Pill V1.0 + ADS1115 + ESP32-C3
% Compatible con BioAmp_EEG_2_0.ino (lineas CSV: SEQ,RAW,EEG,ALFA,ALFARMS,RIEL).
%
% Objetivo: demostracion experimental de EMG superficial de banda LIMITADA.
% Adquisicion Fs=256 Hz -> Nyquist=128 Hz; NO es EMG completo (20-450 Hz).
% NO cambie los nombres de las 6 columnas enviadas por el Arduino.
% Se usa UNICAMENTE RAW (voltaje de salida del BioAmp) para calcular EMG.
% Filtro causal Butterworth 20-90 Hz (orden total 4) + notch 60 Hz Q=30.
% Coeficientes calculados para Fs=256 Hz. No requiere Signal Processing Toolbox.
% RMS deslizante 250 ms, rectificacion y referencia de reposo (2-7 s).
% Los resultados en mV son DE SALIDA del amplificador, NO de la piel.
% Ganancia aparente 3177x fue estimada a 10 Hz; conversion a uV es SOLO
% orientativa y no valida una calibracion EMG en 20-90 Hz.
%
% SEGURIDAD: No colocar electrodos sobre personas conectadas a USB/PC sin
% validacion profesional expresa de seguridad electrica de TODO el sistema.
%
% USO: cierre Monitor Serie de Arduino. Ejecute BioAmp_EMG_Continuo.
% Mantenga el musculo relajado los primeros 8 s. Termine cerrando figura.
% El archivo CSV se escribe continuamente y se cierra al finalizar.

puerto = "COM3";                 % Cambiar si Windows asigna otro COM
Fs = 256;                       % Debe coincidir con el Arduino actual
baud = 230400;
ventanaSeg = 12;
ventanaRMSseg = 0.250;
gananciaAparente10Hz = 3177;    % Estimacion previa; NO es calibracion en banda EMG

fprintf('Puertos disponibles:\n');
disp(serialportlist("available"));
fprintf('EMG experimental 20-90 Hz | Fs=%d Hz | CSV continuo.\n', Fs);
fprintf('Mantener el musculo relajado durante los primeros 8 s.\n');
fprintf('Cierre la ventana de graficas para guardar y detener la captura.\n');
fprintf('AVISO: conversion a uV usa G=%.0f medida a 10 Hz; es aproximada.\n', gananciaAparente10Hz);

% SOS Butterworth de orden 2 (orden global pasabanda 4), Fs=256, 20-90 Hz.
% Cada fila: [b0 b1 b2 a0 a1 a2], con a0=1.
sosBP = [
    0.33738214,  0.67476429,  0.33738214, 1,  0.70284130, 0.33128990;
    1.00000000, -2.00000000,  1.00000000, 1, -1.31580028, 0.53381937
];
% Filtro notch de 60 Hz, Q=30 (banda estrecha); NO resuelve saturaciones.
sosNotch = [0.97603957, -0.19133722, 0.97603957, 1, -0.19133722, 0.95207915];
zBP = zeros(2,2);              % Estado [z1 z2] separado por seccion
zNotch = zeros(1,2);

Nventana = ceil(Fs*ventanaSeg);
Nrms = max(1,round(Fs*ventanaRMSseg));
memoriaCuadrados = zeros(Nrms,1);
sumaCuadrados = 0;

% Memoria de pantalla ACOTADA: el registro completo permanece en disco.
tiemposV = nan(Nventana,1);
rawV = nan(Nventana,1);
emgV = nan(Nventana,1);
rectV = nan(Nventana,1);
rmsV = nan(Nventana,1);
rielV = nan(Nventana,1);

nombre = fullfile(pwd, ['BioAmp_EMG_' datestr(now,'yyyymmdd_HHMMSS') '.csv']);
fid = fopen(nombre,'w');
if fid == -1
    error('No se pudo crear el CSV en %s', pwd);
end
fprintf(fid,'SEQ,tiempo_s,RAW_salida_mV,EMG_20_90_notch60_salida_mV,EMG_rectificada_salida_mV,RMS_250ms_salida_mV,RMS_uV_entrada_APROX_G10Hz,cercaRiel,perdidas_acumuladas\n');

try
    s = serialport(puerto,baud,'Timeout',2);
    configureTerminator(s,'LF');
    flush(s);
catch ME
    fclose(fid);
    rethrow(ME);
end

f = figure('Name','BioAmp | EMG continuo de banda limitada', ...
    'Color',[0.055 0.075 0.12], 'NumberTitle','off', ...
    'Position',[75 55 1150 900]);
t = tiledlayout(f,4,1,'TileSpacing','compact','Padding','compact');
t.Title.String = 'BIOAMP EMG | Reposo inicial 8 s';
t.Title.FontSize = 14;
t.Title.Color = [0.95 0.96 1];

ax1 = nexttile(t);
hRaw = plot(ax1,nan,nan,'Color',[0.95 0.83 0.49]);
ylabel(ax1,'RAW (mV)'); title(ax1,'Salida original ADS A0 (incluye DC e interferencias)');
ax2 = nexttile(t);
hEmg = plot(ax2,nan,nan,'Color',[0.20 0.85 0.90]);
ylabel(ax2,'EMG (mV)'); title(ax2,'EMG limitada 20-90 Hz + notch 60 Hz');
ax3 = nexttile(t);
hRect = plot(ax3,nan,nan,'Color',[0.79 0.56 1]);
ylabel(ax3,'|EMG| (mV)'); title(ax3,'Señal rectificada de onda completa');
ax4 = nexttile(t);
hRMS = plot(ax4,nan,nan,'Color',[1.00 0.73 0.30],'LineWidth',1.7);
hRef = line(ax4,nan,nan,'Color',[0.55 0.7 0.78],'LineStyle','--','LineWidth',1.2);
ylabel(ax4,'RMS (mV)'); xlabel(ax4,'Tiempo de adquisición (s)');
title(ax4,'Envolvente RMS móvil de 250 ms (salida amplificada)');
axs = [ax1 ax2 ax3 ax4];
for k=1:4
    ax = axs(k);
    ax.Color=[0.085 0.11 0.18];
    ax.XColor=[0.65 0.71 0.82];
    ax.YColor=[0.65 0.71 0.82];
    ax.GridColor=[0.40 0.48 0.60];
    ax.GridAlpha=0.22;
    ax.Title.Color=[0.91 0.93 0.98];
    ax.FontSize=10;
    grid(ax,'on');
    xlim(ax,[0 ventanaSeg]);
end

n=0;
primeraSeq=nan;
ultimaSeq=nan;
primeraRAW=nan;
perdidas=0;
fallosConsecutivos=0;
baseline=nan;
reposoMuestras=zeros(5*Fs,1);
nReposo=0;

try
    while isgraphics(f)
        try
            linea = strtrim(char(readline(s)));
            fallosConsecutivos=0;
        catch ME
            if ~isgraphics(f), break; end
            fallosConsecutivos=fallosConsecutivos+1;
            fprintf('Lectura interrumpida (%d/5): %s\n',fallosConsecutivos,ME.message);
            if fallosConsecutivos>=5
                warning('Se detuvo por cinco errores consecutivos del puerto.');
                break;
            end
            drawnow limitrate;
            continue;
        end

        % Datos: SEQ, RAW_mV, EEG_mV, Alfa_mV, AlfaRMS_mV, RIEL
        v = sscanf(linea,'%f,%f,%f,%f,%f,%f');
        if numel(v)~=6 || any(~isfinite(v)) || v(1)<0 || v(1)~=floor(v(1))
            continue;
        end
        if ~isnan(ultimaSeq) && v(1)<=ultimaSeq
            warning('SEQ retrocedio: reinicio del ESP32 o puerto equivocado.');
            break;
        end
        if ~isnan(ultimaSeq)
            perdidas=perdidas+max(0,v(1)-ultimaSeq-1);
        end
        ultimaSeq=v(1);
        if isnan(primeraSeq), primeraSeq=v(1); end
        tm=(v(1)-primeraSeq)/Fs;
        raw_mV=v(2);
        cercaRiel=double(v(6)~=0);
        if isnan(primeraRAW), primeraRAW=raw_mV; end
        x=raw_mV-primeraRAW;  % Quitar DC; tambien lo elimina el pasabanda.

        % Dos secciones IIR con forma directa II transpuesta.
        for j=1:2
            b=sosBP(j,:);
            y=b(1)*x+zBP(j,1);
            zBP(j,1)=b(2)*x-b(5)*y+zBP(j,2);
            zBP(j,2)=b(3)*x-b(6)*y;
            x=y;
        end
        b=sosNotch;
        y=b(1)*x+zNotch(1);
        zNotch(1)=b(2)*x-b(5)*y+zNotch(2);
        zNotch(2)=b(3)*x-b(6)*y;
        emg_mV=y;
        emgRect_mV=abs(emg_mV);

        % RMS movil de 250 ms, sin necesidad de toolbox.
        n=n+1;
        p=mod(n-1,Nrms)+1;
        sumaCuadrados=sumaCuadrados-memoriaCuadrados(p);
        memoriaCuadrados(p)=emg_mV^2;
        sumaCuadrados=sumaCuadrados+memoriaCuadrados(p);
        emgRMS_mV=sqrt(max(sumaCuadrados,0)/min(n,Nrms));
        emgEntrada_uV_APROX=emgRMS_mV*1000/gananciaAparente10Hz;

        % Referencia relativa: primeros 2-7 s, SOLO si realmente hay reposo.
        if tm>=2 && tm<7 && cercaRiel==0 && nReposo<numel(reposoMuestras)
            nReposo=nReposo+1;
            reposoMuestras(nReposo)=emgRMS_mV;
        end
        if isnan(baseline) && tm>=7 && nReposo>=3*Fs
            baseline=median(reposoMuestras(1:nReposo));
            fprintf('Referencia de reposo (2-7 s): %.3f mV RMS. NO es MVC.\n',baseline);
        end

        % Archivo de disco: escribe CADA muestra, sin limite de duracion.
        fprintf(fid,'%d,%.6f,%.5f,%.5f,%.5f,%.5f,%.4f,%d,%d\n', ...
            round(v(1)),tm,raw_mV,emg_mV,emgRect_mV,emgRMS_mV, ...
            emgEntrada_uV_APROX,cercaRiel,round(perdidas));

        % Guardar unicamente los ultimos 12 s en memoria para la figura.
        p=mod(n-1,Nventana)+1;
        tiemposV(p)=tm;
        rawV(p)=raw_mV;
        emgV(p)=emg_mV;
        rectV(p)=emgRect_mV;
        rmsV(p)=emgRMS_mV;
        rielV(p)=cercaRiel;

        if mod(n,26)==0 && isgraphics(f)
            idx=find(isfinite(tiemposV));
            [ts,orden]=sort(tiemposV(idx));
            ix=idx(orden);
            rv=rawV(ix); ev=emgV(ix); av=rectV(ix); mv=rmsV(ix);
            set(hRaw,'XData',ts,'YData',rv);
            set(hEmg,'XData',ts,'YData',ev);
            set(hRect,'XData',ts,'YData',av);
            set(hRMS,'XData',ts,'YData',mv);
            ini=max(0,tm-ventanaSeg);
            for k=1:4, xlim(axs(k),[ini ini+ventanaSeg]); end
            margen=max(25,0.1*(max(rv)-min(rv)));
            ylim(ax1,[min(rv)-margen max(rv)+margen]);
            escalaEMG=max(1,1.2*max(abs(ev)));
            ylim(ax2,[-escalaEMG escalaEMG]);
            ylim(ax3,[0 max(1,1.2*max(av))]);
            maxRMS=max(1,1.25*max([mv;baseline(isfinite(baseline))]));
            ylim(ax4,[0 maxRMS]);
            if isfinite(baseline)
                set(hRef,'XData',[ini ini+ventanaSeg],'YData',[baseline baseline]);
            end

            % Medicion de 60 Hz sobre la señal cruda, antes del notch.
            sel60=ts>=tm-2;
            amp60=nan;
            if nnz(sel60)>=Fs
                r60=rv(sel60)-mean(rv(sel60));
                amp60=2*abs(sum(r60.*exp(-1i*2*pi*60*ts(sel60))))/nnz(sel60);
            end
            rielVisible=sum(rielV(ix));
            estado='REPOSO INICIAL';
            if tm>=8, estado='REGISTRO CONTINUO'; end
            if isfinite(baseline) && baseline>0
                relacion=sprintf(' | RMS/reposo: %.2fx',emgRMS_mV/baseline);
            else
                relacion=' | RMS/reposo: pendiente';
            end
            aviso='';
            if rielVisible>0, aviso=' | AVISO: muestras cerca de riel'; end
            t.Title.String=sprintf('%s | %.1f s | RMS %.2f mV | 60 Hz %.1f mV pico | perdidas %d%s%s', ...
                estado,tm,emgRMS_mV,amp60,perdidas,relacion,aviso);
            drawnow limitrate;
        end
    end
catch ME
    warning('Registro interrumpido: %s',ME.message);
end

% Al cerrar figura, cerrar archivo para asegurar los datos.
fclose(fid);
clear s;
fprintf('\nEMG finalizado. %d muestras, %.1f s, %d perdidas.\n',n, ...
    max(0,(ultimaSeq-primeraSeq)/Fs),perdidas);
fprintf('Archivo CSV guardado en:\n%s\n',nombre);
fprintf('RMS uV es SOLO estimacion con ganancia medida a 10Hz.\n');
end
