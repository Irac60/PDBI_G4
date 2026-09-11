# Proyecto de Biodiseño - Grupo 4
| Integrantes                     | Código de estudiante |
|---------------------------------|:--------------------:|
| Jean Pierre Roy Cortez Alban    | 76567780             |
| Abigail Valentina Vasco Panduro | 72876106             |
| Naun Aldair Cari Quispe         | 60758913             |
| César Roberto García Bonilla    | 72005834             |
| Rodrigo Emiliano Salas Cano     | 61010375             |

<img width="1280" height="960" alt="WhatsApp Image 2026-09-04 at 09 41 41" src="https://github.com/user-attachments/assets/6244565d-8e03-4fca-9765-309f5e5672a5" />

# Cronograma del proyecto

## Sistema de monitorización nocturna de crisis tonicoclónicas

### Actividades y horas estimadas

| N.° | Actividad | Semanas | Horas |
|---:|---|:---:|---:|
| 1 | Definición y delimitación de la problemática | 1 | 5 |
| 2 | Elaboración del plan de trabajo | 1–2 | 4 |
| 3 | Revisión bibliográfica y evidencia clínica | 1–3 | 8 |
| 4 | Análisis del estado de la tecnología | 1–3 | 8 |
| 5 | Definición de requerimientos del sistema | 3–4 | 6 |
| 6 | Diseño conceptual de la solución | 4–5 | 6 |
| 7 | Selección de tecnologías y hardware | 4–5 | 7 |
| 8 | Diseño de la arquitectura del sistema | 5–6 | 9 |
| 9 | Desarrollo e integración del hardware | 6–9 | 12 |
| 10 | Adquisición y registro de señales | 7–9 | 10 |
| 11 | Procesamiento de señales | 8–10 | 10 |
| 12 | Desarrollo del algoritmo de detección | 9–12 | 14 |
| 13 | Desarrollo del sistema de alertas | 10–12 | 8 |
| 14 | Integración del prototipo | 11–13 | 12 |
| 15 | Pruebas técnicas del sistema | 12–14 | 10 |
| 16 | Evaluación del desempeño y falsas alarmas | 13–14 | 8 |
| 17 | Optimización del prototipo | 14–15 | 8 |
| 18 | Documentación y presentación final | 13–15 | 6 |
| **Total** | | | **151 h** |

## Diagrama de Gantt


```mermaid
gantt
    title Cronograma del proyecto
    dateFormat YYYY-MM-DD
    axisFormat         Semana %W　　　　　　　　　　　　
    tickInterval 1week
    todayMarker off

    section Investigación
    Problemática                         :a1, 2026-01-05, 7d
    Plan de trabajo                      :a2, 2026-01-05, 14d
    Revisión bibliográfica               :a3, 2026-01-05, 21d
    Estado tecnológico                   :a4, 2026-01-05, 21d

    section Diseño
    Requerimientos                       :b1, 2026-01-19, 14d
    Diseño conceptual                    :b2, 2026-01-26, 14d
    Tecnología y hardware                :b3, 2026-01-26, 14d
    Arquitectura del sistema             :b4, 2026-02-02, 14d

    section Desarrollo
    Integración de hardware              :c1, 2026-02-09, 28d
    Registro de señales                  :c2, 2026-02-16, 21d
    Procesamiento de señales             :c3, 2026-02-23, 21d
    Algoritmo de detección               :c4, 2026-03-02, 28d

    section Alerta e integración
    Sistema de alertas                   :d1, 2026-03-09, 21d
    Integración del prototipo            :d2, 2026-03-16, 21d

    section Validación
    Pruebas técnicas                     :e1, 2026-03-23, 21d
    Desempeño y falsas alarmas           :e2, 2026-03-30, 14d
    Optimización                         :e3, 2026-04-06, 14d

    section Cierre
    Documentación final                  :f1, 2026-03-30, 21d
```
