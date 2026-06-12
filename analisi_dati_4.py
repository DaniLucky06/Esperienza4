import sys, os
dire = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dire, "analisi-dati-python"))
from analysis import *
import matplotlib.pyplot as plt
import numpy as np

g = 9.807
rho = 1000.
offset_temperature = 0.

dati = np.loadtxt(os.path.join(dire, 'dati_acqua.csv'), delimiter=',')
rimuovi_da_zero = 0
fine_discesa = 16 - rimuovi_da_zero
temperature_discesa = dati[0:fine_discesa, 0] + 273.15 - offset_temperature
altezze_discesa = dati[0:fine_discesa, 1] / 100
pressioni_discesa = dati[0:fine_discesa, 2] * 100
inizio_salita = 15 + rimuovi_da_zero
temperature_salita = dati[inizio_salita:, 0] + 273.15 - offset_temperature
altezze_salita = dati[inizio_salita:, 1] / 100
pressioni_salita = dati[inizio_salita:, 2] * 100

altezza_dx = 98. / 100
dew_point = 8. + 273.15

dh_discesa = altezze_discesa - altezza_dx
dh_salita = altezze_salita - altezza_dx

p_int_discesa = pressioni_discesa + rho * g * dh_discesa
p_int_salita = pressioni_salita + rho * g * dh_salita

ris_temperatura = 0.01
ris_h = 2. / 1000
ris_p_ext = 0.01 * 100
err_t = ris_temperatura / np.sqrt(12)
err_h = ris_h / np.sqrt(12) * np.sqrt(2)
err_p_ext = ris_p_ext / np.sqrt(12)

err_p_int = np.sqrt(err_p_ext**2 + (rho * g * err_h)**2)

# ==============================================================================
# PARTE 1: ANALISI CON TUTTI I PUNTI SPERIMENTALI
# ==============================================================================

uni_discesa = np.ones(len(temperature_discesa))
misure_discesa = WeightedMeasurements(temperature_discesa, p_int_discesa, uni_discesa * err_t, uni_discesa * err_p_int)
reg_data_discesa, xy_data_discesa = weightedLinReg(misure_discesa)

uni_salita = np.ones(len(temperature_salita))
misure_salita = WeightedMeasurements(temperature_salita, p_int_salita, uni_salita * err_t, uni_salita * err_p_int)
reg_data_salita, xy_data_salita = weightedLinReg(misure_salita)

# 1a. Grafico dei Fit (Tutti i punti)
plotGraphs([xy_data_discesa,misure_discesa, xy_data_salita, misure_salita], GraphVisuals(['k', 'rx', 'b', 'gx']))

print("======================================================================")
print("===                  ANALISI COMPLETA (TUTTI I PUNTI)              ===")
print("======================================================================")
print("Chi2 ridotto discesa:", reg_data_discesa.chi2/reg_data_discesa.nu)
print("Chi2 ridotto salita :", reg_data_salita.chi2/reg_data_salita.nu)
print(f"a discesa: {reg_data_discesa.a} | b discesa: {reg_data_discesa.b}")
print(f"a salita : {reg_data_salita.a} | b salita : {reg_data_salita.b}")
print(f"a_err discesa: {reg_data_discesa.a_err} | b_err discesa: {reg_data_discesa.b_err}")
print(f"a_err salita : {reg_data_salita.a_err} | b_err salita : {reg_data_salita.b_err}")

t_a = np.abs(reg_data_discesa.a - reg_data_salita.a) / np.sqrt(reg_data_discesa.a_err**2 + reg_data_salita.a_err**2)
t_b = np.abs(reg_data_discesa.b - reg_data_salita.b) / np.sqrt(reg_data_discesa.b_err**2 + reg_data_salita.b_err**2)
print(f"T-test completo (a, b): {t_a} | {t_b}")

# T-test dei parametri 'a' rispetto a 0
t_a_zero_discesa = np.abs(reg_data_discesa.a) / reg_data_discesa.a_err
t_a_zero_salita = np.abs(reg_data_salita.a) / reg_data_salita.a_err
print(f"T-test intercetta con lo zero (discesa, salita): {t_a_zero_discesa} | {t_a_zero_salita}\n")

# 1b. Grafico dei Residui (Tutti i punti)
p_predetta_discesa = reg_data_discesa.a + reg_data_discesa.b * temperature_discesa
p_predetta_salita = reg_data_salita.a + reg_data_salita.b * temperature_salita
residui_discesa = p_int_discesa - p_predetta_discesa
residui_salita = p_int_salita - p_predetta_salita

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.6)
ax1.errorbar(temperature_discesa, residui_discesa, yerr=uni_discesa*err_p_int, xerr=uni_discesa*err_t, fmt='rx', ecolor='r', capsize=2, label='Residui Discesa')
ax1.set_title('Residui Completi - Discesa')
ax1.set_xlabel('Temperatura T (K)')
ax1.set_ylabel('Residui $\Delta P$ (Pa)')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend()

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.6)
ax2.errorbar(temperature_salita, residui_salita, yerr=uni_salita*err_p_int, xerr=uni_salita*err_t, fmt='gx', ecolor='g', capsize=2, label='Residui Salita')
ax2.set_title('Residui Completi - Salita')
ax2.set_xlabel('Temperatura T (K)')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# PARTE 2: FILTRAGGIO - ESCLUSIONE DEI PRIMI 4 PUNTI PIÙ VICINI A 0°C
# ==============================================================================

# Identificazione ed esclusione dei 4 valori di temperatura minori
idx_escludi_discesa = np.argsort(temperature_discesa)[:4]
mask_filtrata_discesa = np.ones(len(temperature_discesa), dtype=bool)
mask_filtrata_discesa[idx_escludi_discesa] = False
t_discesa_f = temperature_discesa[mask_filtrata_discesa]
p_discesa_f = p_int_discesa[mask_filtrata_discesa]

idx_escludi_salita = np.argsort(temperature_salita)[:4]
mask_filtrata_salita = np.ones(len(temperature_salita), dtype=bool)
mask_filtrata_salita[idx_escludi_salita] = False
t_salita_f = temperature_salita[mask_filtrata_salita]
p_salita_f = p_int_salita[mask_filtrata_salita]

# Calcolo nuovi fit lineari pesati sui dati filtrati
uni_discesa_f = np.ones(len(t_discesa_f))
misure_discesa_f = WeightedMeasurements(t_discesa_f, p_discesa_f, uni_discesa_f * err_t, uni_discesa_f * err_p_int)
reg_data_discesa_f, xy_data_discesa_f = weightedLinReg(misure_discesa_f)

uni_salita_f = np.ones(len(t_salita_f))
misure_salita_f = WeightedMeasurements(t_salita_f, p_salita_f, uni_salita_f * err_t, uni_salita_f * err_p_int)
reg_data_salita_f, xy_data_salita_f = weightedLinReg(misure_salita_f)

# 2a. Grafico dei Fit (Senza i 4 punti freddi)
plotGraphs([xy_data_discesa_f, misure_discesa_f, xy_data_salita_f, misure_salita_f], GraphVisuals(['k', 'ro', 'b', 'go']))

print("======================================================================")
print("===             ANALISI FILTRATA (SENZA I 4 PUNTI PIÙ FREDDI)      ===")
print("======================================================================")
print("Chi2 ridotto discesa filtrata:", reg_data_discesa_f.chi2 / reg_data_discesa_f.nu)
print("Chi2 ridotto salita filtrata :", reg_data_salita_f.chi2 / reg_data_salita_f.nu)
print(f"a discesa: {reg_data_discesa_f.a} | b discesa: {reg_data_discesa_f.b}")
print(f"a salita : {reg_data_salita_f.a} | b salita : {reg_data_salita_f.b}")
print(f"a_err discesa: {reg_data_discesa_f.a_err} | b_err discesa: {reg_data_discesa_f.b_err}")
print(f"a_err salita : {reg_data_salita_f.a_err} | b_err salita : {reg_data_salita_f.b_err}")

t_a_f = np.abs(reg_data_discesa_f.a - reg_data_salita_f.a) / np.sqrt(reg_data_discesa_f.a_err**2 + reg_data_salita_f.a_err**2)
t_b_f = np.abs(reg_data_discesa_f.b - reg_data_salita_f.b) / np.sqrt(reg_data_discesa_f.b_err**2 + reg_data_salita_f.b_err**2)
print(f"Nuovo T-test filtrato (a, b): {t_a_f} | {t_b_f}")

# T-test dei parametri 'a' filtrati rispetto a 0
t_a_zero_discesa_f = np.abs(reg_data_discesa_f.a) / reg_data_discesa_f.a_err
t_a_zero_salita_f = np.abs(reg_data_salita_f.a) / reg_data_salita_f.a_err
print(f"Nuovo T-test intercetta con lo zero (discesa, salita): {t_a_zero_discesa_f} | {t_a_zero_salita_f}\n")

# 2b. Grafico dei Residui (Senza i 4 punti freddi)
p_pred_discesa_f = reg_data_discesa_f.a + reg_data_discesa_f.b * t_discesa_f
p_pred_salita_f = reg_data_salita_f.a + reg_data_salita_f.b * t_salita_f
residui_discesa_f = p_discesa_f - p_pred_discesa_f
residui_salita_f = p_salita_f - p_pred_salita_f

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.6)
ax3.errorbar(t_discesa_f, residui_discesa_f, yerr=uni_discesa_f*err_p_int, xerr=uni_discesa_f*err_t, fmt='rx', ecolor='r', capsize=2, label='Residui Discesa Filtrata')
ax3.set_title('Residui Filtrati - Discesa')
ax3.set_xlabel('Temperatura T (K)')
ax3.set_ylabel('Residui $\Delta P$ (Pa)')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend()

ax4.axhline(y=0, color='k', linestyle='--', alpha=0.6)
ax4.errorbar(t_salita_f, residui_salita_f, yerr=uni_salita_f*err_p_int, xerr=uni_salita_f*err_t, fmt='gx', ecolor='g', capsize=2, label='Residui Salita Filtrata')
ax4.set_title('Residui Filtrati - Salita')
ax4.set_xlabel('Temperatura T (K)')
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.legend()
plt.tight_layout()
plt.show()