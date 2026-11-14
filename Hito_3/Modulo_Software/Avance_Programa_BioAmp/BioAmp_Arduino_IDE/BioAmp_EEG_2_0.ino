
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
