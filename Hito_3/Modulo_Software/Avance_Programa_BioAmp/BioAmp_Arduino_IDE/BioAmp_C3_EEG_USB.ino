/*
  BIOAMP EEG 2.0 - BioAmp EXG Pill V1.0 + ADS1115 + ESP32-C3
  Frecuencia nominal: 256 muestras por segundo, A0 con ADS1115.
  Salida CSV serie: secuencia,RAW_mV,EEG_mV,Alfa_mV,AlfaRMS_mV,RIEL
  Uso exclusivo junto con BioAmp_EEG_2_0.m (230400 baudios).

  CONEXIONES DE ADQUISICION (sin generar ondas artificiales):
  BioAmp: VCC=3V3, GND=GND, OUT=ADS A0.
  ADS1115: VDD=3V3, GND=GND, ADDR=GND,
           SDA=GPIO8, SCL=GPIO9.
  NO conectar el generador GPIO6, la rama de 1 Mohm ni ADS A1.
  Los electrodos, si se utilizan, van SOLO a entradas BioAmp,
  no a la entrada ADS1115 directamente.

  SEGURIDAD: No conectar personas mientras USB/PC u otra alimentacion
  formen parte del sistema sin validacion especifica y documentada de
  aislamiento, fuga y seguridad de TODO el montaje en esa configuracion.
  Esto es un prototipo experimental, no dispositivo medico.

  Se mantienen exactamente los coeficientes de filtros ya ensayados:
  0.5-29.5 Hz adaptado de Upside Down Labs, BioAmp EXG Pill
  (https://github.com/upsidedownlabs/BioAmp-EXG-Pill, licencia MIT).
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
  8-13 Hz: Butterworth por dos secciones de segundo orden.
*/
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <math.h>

Adafruit_ADS1115 ads;
constexpr uint32_t FS = 256;
constexpr uint32_t BAUDIOS = 230400;
constexpr uint32_t PASO_US = 1000000UL / FS; // 3906
constexpr uint16_t RESTO_US = 1000000UL % FS; // 64
constexpr uint16_t VENTANA = FS;
uint32_t siguiente_us = 0;
uint16_t fraccion = 0;
uint32_t secuencia = 0;
float referencia_mV = 0.0f;
bool primera_muestra = true;

float filtroEEG(float entrada) {
  float y = entrada;
  {
    static float z1=0, z2=0;
    float x=y+0.95391350f*z1-0.25311356f*z2;
    y=0.00735282f*x+0.01470564f*z1+0.00735282f*z2;
    z2=z1; z1=x;
  }
  {
    static float z1=0, z2=0;
    float x=y+1.20596630f*z1-0.60558332f*z2;
    y=x+2.0f*z1+z2;
    z2=z1; z1=x;
  }
  {
    static float z1=0, z2=0;
    float x=y+1.97690645f*z1-0.97706395f*z2;
    y=x-2.0f*z1+z2;
    z2=z1; z1=x;
  }
  {
    static float z1=0, z2=0;
    float x=y+1.99071687f*z1-0.99086813f*z2;
    y=x-2.0f*z1+z2;
    z2=z1; z1=x;
  }
  return y;
}

struct Biquad {
  float b0,b1,b2,a1,a2;
  float z1=0,z2=0;
  float procesar(float x) {
    float y=b0*x+z1;
    z1=b1*x-a1*y+z2;
    z2=b2*x-a2*y;
    return y;
  }
};

Biquad alfa1={0.003460684761f,0.006921369523f,0.003460684761f,
              -1.819401294431f,0.903462400151f};
Biquad alfa2={1.0f,-2.0f,1.0f,-1.888033255808f,0.930504523419f};

float potencias[VENTANA]={0};
uint16_t indice=0, cantidad=0;
float suma_cuadrados=0;
float rmsAlfa(float alfa) {
  float p=alfa*alfa;
  suma_cuadrados-=potencias[indice];
  potencias[indice]=p;
  suma_cuadrados+=p;
  indice=(indice+1)%VENTANA;
  if(cantidad<VENTANA) ++cantidad;
  return sqrtf(fmaxf(0.0f,suma_cuadrados)/cantidad);
}

void avanzarReloj() {
  siguiente_us += PASO_US;
  fraccion += RESTO_US;
  if (fraccion >= FS) {
    fraccion -= FS;
    ++siguiente_us;
  }
}

void setup() {
  Serial.begin(BAUDIOS);
  delay(1000);
  Wire.begin(8,9);
  Wire.setClock(400000);
  if(!ads.begin(0x48,&Wire)) {
    while(true) {
      Serial.println("ERROR: ADS1115 ausente en 0x48");
      delay(1000);
    }
  }
  ads.setGain(GAIN_ONE);
  ads.setDataRate(RATE_ADS1115_860SPS);
  siguiente_us=micros();
  Serial.println("SEQ,RAW_mV,EEG_mV,Alfa_mV,AlfaRMS_mV,RIEL");
}

void loop() {
  uint32_t ahora=micros();
  if((int32_t)(ahora-siguiente_us)<0) return;

  // Evitar acumulacion de atraso excepcional.
  if((uint32_t)(ahora-siguiente_us)>2*PASO_US) {
    siguiente_us=ahora;
    fraccion=0;
  }
  avanzarReloj();

  int16_t cuentas=ads.readADC_SingleEnded(0);
  float raw_mV=1000.0f*ads.computeVolts(cuentas);
  if(primera_muestra) {
    referencia_mV=raw_mV;
    primera_muestra=false;
  }
  float eeg_mV=filtroEEG(raw_mV-referencia_mV);
  float alfa_mV=alfa2.procesar(alfa1.procesar(eeg_mV));
  float alfa_rms_mV=rmsAlfa(alfa_mV);

  // RIEL=1 es aviso de proximidad al rango alimentado, no
  // prueba de saturacion analogica del BioAmp.
  uint8_t riel=(raw_mV<100.0f || raw_mV>3200.0f) ? 1 : 0;
  Serial.print(secuencia++); Serial.write(',');
  Serial.print(raw_mV,3); Serial.write(',');
  Serial.print(eeg_mV,3); Serial.write(',');
  Serial.print(alfa_mV,3); Serial.write(',');
  Serial.print(alfa_rms_mV,3); Serial.write(',');
  Serial.println(riel);
}
