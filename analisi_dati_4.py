import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Configurazione path per importare la libreria personalizzata
dire = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dire, "analisi-dati-python"))
from analysis import *

# ==============================================================================
# CONFIGURAZIONE DIMENSIONE FONT E PARAMETRI GRAFICI (rcParams)
# ==============================================================================
plt.rcParams.update({
    'font.size': 17,          # Dimensione generale del testo e valori numerici assi
    'axes.titlesize': 20,     # Dimensione del titolo dei singoli grafici
    'axes.labelsize': 18,     # Dimensione delle etichette degli assi (X e Y)
    'legend.fontsize': 15,    # Dimensione del testo all'interno delle legende
    'legend.loc': 'lower right'      # Posizionamento automatico ottimale della legenda
})

# ==============================================================================
# COSTANTI E PARAMETRI FISICI
# ==============================================================================
g = 9.807          # Accelerazione di gravità [m/s^2]
rho = 1000.        # Densità dell'acqua [kg/m^3]
altezza_dx = 0.98  # Altezza del ramo di destra (riferimento) [m]
T_0 = 273.15       # Zero gradi Celsius in Kelvin [K]

# Parametri per le correzioni sistematiche (estratti da "Esperienza 4 - Volumi")
V0 = 611.92        # Volume bottiglia [cm^3]
v = 7.315          # Volume tubino + tubino vetro non termalizzati [cm^3]
gamma = 1.5e-5     # Coefficiente di dilatazione volumica del Pyrex [K^-1]
T_a = 293.15       # Temperatura ambiente stimata (20°C) [K]

# ==============================================================================
# CARICAMENTO E PREPARAZIONE DATI
# ==============================================================================
dati = np.loadtxt(os.path.join(dire, 'dati_acqua.csv'), delimiter=',')

# Definizione dell'offset termico da sottrarre
offset_temperatura = 0.0 # 0.09

# Separazione dati Discesa (prime 16 righe) e Salita (restanti)
fine_discesa = 16
inizio_salita = 15
filtro = 6

# Discesa: Applicazione offset, conversione in Kelvin, metri e Pascal
t_discesa = (dati[0:fine_discesa, 0] - offset_temperatura) + T_0
h_discesa = dati[0:fine_discesa, 1] / 100
p_ext_discesa = dati[0:fine_discesa, 2] * 100

# Salita: Applicazione offset, conversione in Kelvin, metri e Pascal
t_salita = (dati[inizio_salita:, 0] - offset_temperatura) + T_0
h_salita = dati[inizio_salita:, 1] / 100
p_ext_salita = dati[inizio_salita:, 2] * 100

# Calcolo dislivelli e pressione interna assoluta (P_ext + rho*g*dh)
dh_discesa = h_discesa - altezza_dx
dh_salita = h_salita - altezza_dx

p_int_discesa = p_ext_discesa + rho * g * dh_discesa
p_int_salita = p_ext_salita + rho * g * dh_salita

# ==============================================================================
# INCERTEZZE E PROPAGAZIONE DEGLI ERRORI
# ==============================================================================
# Risoluzioni strumentali
ris_temperatura = 0.01       # [K]
ris_h = 0.002                # 2 mm [m]
ris_p_ext = 1.0              # 0.01 mbar -> 1 Pa [Pa]

# Incertezze tipo B (distribuzione uniforme -> diviso radice di 12)
err_t = ris_temperatura / np.sqrt(12)
err_h = (ris_h / np.sqrt(12)) * np.sqrt(2) # x rad(2) per propagazione differenza di due altezze
err_p_ext = ris_p_ext / np.sqrt(12)

# Errore propagato sulla pressione interna (somma in quadratura)
err_p_int_val = np.sqrt(err_p_ext**2 + (rho * g * err_h)**2)

# Vettori di errori omogenei per sfruttare la libreria analysis.py
err_p_discesa = np.ones(len(t_discesa)) * err_p_int_val
err_p_salita  = np.ones(len(t_salita)) * err_p_int_val
err_t_discesa = np.ones(len(t_discesa)) * err_t
err_t_salita  = np.ones(len(t_salita)) * err_t


# --- FUNZIONE DI SUPPORTO PER EVITARE CODICE RIPETUTO ---
def calcola_zero_assoluto(reg_data):
    """Calcola lo zero assoluto (T0 = -a/b) e propaga l'errore dai parametri di fit."""
    T_zero = -reg_data.a / reg_data.b
    err_T_zero = np.abs(T_zero) * np.sqrt((reg_data.a_err / reg_data.a)**2 + (reg_data.b_err / reg_data.b)**2)
    return T_zero, err_T_zero


# ==============================================================================
# PARTE 1: ANALISI COMPLETA (TUTTI I PUNTI SPERIMENTALI)
# ==============================================================================
print("\n" + "="*70 + "\n=== 1. ANALISI COMPLETA (TUTTI I PUNTI)\n" + "="*70)

# Fit pesati
mis_discesa = WeightedMeasurements(t_discesa, p_int_discesa, err_t_discesa, err_p_discesa)
reg_discesa, xy_discesa = weightedLinReg(mis_discesa)

mis_salita = WeightedMeasurements(t_salita, p_int_salita, err_t_salita, err_p_salita)
reg_salita, xy_salita = weightedLinReg(mis_salita)

# Stampa metriche di bontà del fit
print(f"Chi2 ridotto discesa : {reg_discesa.chi2/reg_discesa.nu:.2f}")
print(f"Chi2 ridotto salita  : {reg_salita.chi2/reg_salita.nu:.2f}\n")

# Calcolo zero assoluto
T0_disc, err_T0_disc = calcola_zero_assoluto(reg_discesa)
T0_sal, err_T0_sal = calcola_zero_assoluto(reg_salita)

print(f"Zero assoluto (Discesa): {T0_disc:.2f} +/- {err_T0_disc:.2f} K  ({T0_disc - T_0:.2f} °C)")
print(f"Zero assoluto (Salita) : {T0_sal:.2f} +/- {err_T0_sal:.2f} K  ({T0_sal - T_0:.2f} °C)")
print(f"Test compatibilità 0 K : {np.abs(T0_disc)/err_T0_disc:.2f} (disc) | {np.abs(T0_sal)/err_T0_sal:.2f} (sal)\n")

# Grafico fit e Residui
print("=> Mostro i grafici del Fit Completo (chiudi la finestra per proseguire)")
plt.figure(10, figsize=(10, 6))
plt.title('Fit Completo: Pressione vs Temperatura')
plt.xlabel('Temperatura T (K)')
plt.ylabel('Pressione $P$ (Pa)')
plt.grid(True, linestyle=':', alpha=0.5)
# Generazione fittizia per la legenda
plt.plot([], [], 'rx', label='Dati Discesa')
plt.plot([], [], 'k-', label='Fit Discesa')
plt.plot([], [], 'gx', label='Dati Salita')
plt.plot([], [], 'b-', label='Fit Salita')
plt.legend()
plotGraphs([xy_discesa, mis_discesa, xy_salita, mis_salita], GraphVisuals(['k', 'rx', 'b', 'gx']), figureID=10)

residui_disc = p_int_discesa - (reg_discesa.a + reg_discesa.b * t_discesa)
residui_sal = p_int_salita - (reg_salita.a + reg_salita.b * t_salita)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
ax1.errorbar(t_discesa, residui_disc, yerr=err_p_discesa, xerr=err_t_discesa, fmt='rx', ecolor='r', capsize=2, label='Dati Sperimentali')
ax1.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax1.set(title='Residui Completi - Discesa', xlabel='Temperatura T (K)', ylabel='Residui $\Delta P$ (Pa)')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend()

ax2.errorbar(t_salita, residui_sal, yerr=err_p_salita, xerr=err_t_salita, fmt='gx', ecolor='g', capsize=2, label='Dati Sperimentali')
ax2.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax2.set(title='Residui Completi - Salita', xlabel='Temperatura T (K)')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# PARTE 2: FILTRAGGIO (ESCLUSIONE CONDENSA - PRIMI {filtro} PUNTI PIÙ FREDDI)
# ==============================================================================
print("\n" + "="*70 + "\n=== 2. ANALISI FILTRATA (SENZA CONDENSA)\n" + "="*70)

# Mascheratura dati
mask_disc = np.ones(len(t_discesa), dtype=bool)
mask_disc[np.argsort(t_discesa)[:filtro]] = False
t_disc_f, p_disc_f, err_t_disc_f, err_p_disc_f = t_discesa[mask_disc], p_int_discesa[mask_disc], err_t_discesa[mask_disc], err_p_discesa[mask_disc]

mask_sal = np.ones(len(t_salita), dtype=bool)
mask_sal[np.argsort(t_salita)[:filtro]] = False
t_sal_f, p_sal_f, err_t_sal_f, err_p_sal_f = t_salita[mask_sal], p_int_salita[mask_sal], err_t_salita[mask_sal], err_p_salita[mask_sal]

# Fit pesati sui dati filtrati
mis_disc_f = WeightedMeasurements(t_disc_f, p_disc_f, err_t_disc_f, err_p_disc_f)
reg_disc_f, xy_disc_f = weightedLinReg(mis_disc_f)

mis_sal_f = WeightedMeasurements(t_sal_f, p_sal_f, err_t_sal_f, err_p_sal_f)
reg_sal_f, xy_sal_f = weightedLinReg(mis_sal_f)

# Stampa metriche e risultati
print(f"Chi2 ridotto discesa filtrata : {reg_disc_f.chi2/reg_disc_f.nu:.2f}")
print(f"Chi2 ridotto salita filtrata  : {reg_sal_f.chi2/reg_sal_f.nu:.2f}\n")

T0_disc_f, err_T0_disc_f = calcola_zero_assoluto(reg_disc_f)
T0_sal_f, err_T0_sal_f = calcola_zero_assoluto(reg_sal_f)

print(f"Zero assoluto (Discesa): {T0_disc_f:.2f} +/- {err_T0_disc_f:.2f} K  ({T0_disc_f - T_0:.2f} °C)")
print(f"Zero assoluto (Salita) : {T0_sal_f:.2f} +/- {err_T0_sal_f:.2f} K  ({T0_sal_f - T_0:.2f} °C)")
print(f"Test compatibilità 0 K : {np.abs(T0_disc_f)/err_T0_disc_f:.2f} (disc) | {np.abs(T0_sal_f)/err_T0_sal_f:.2f} (sal)\n")

# Grafico fit e Residui Filtrati
print("=> Mostro i grafici del Fit Filtrato")
plt.figure(20, figsize=(10, 6))
plt.title('Fit Filtrato: Pressione vs Temperatura')
plt.xlabel('Temperatura T (K)')
plt.ylabel('Pressione $P$ (Pa)')
plt.grid(True, linestyle=':', alpha=0.5)
# Generazione fittizia per la legenda
plt.plot([], [], 'ro', label='Dati Filtrati Discesa')
plt.plot([], [], 'k-', label='Fit Discesa')
plt.plot([], [], 'go', label='Dati Filtrati Salita')
plt.plot([], [], 'b-', label='Fit Salita')
plt.legend()
plotGraphs([xy_disc_f, mis_disc_f, xy_sal_f, mis_sal_f], GraphVisuals(['k', 'ro', 'b', 'go']), figureID=20)

residui_disc_f = p_disc_f - (reg_disc_f.a + reg_disc_f.b * t_disc_f)
residui_sal_f = p_sal_f - (reg_sal_f.a + reg_sal_f.b * t_sal_f)

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
ax3.errorbar(t_disc_f, residui_disc_f, yerr=err_p_disc_f, xerr=err_t_disc_f, fmt='rx', ecolor='r', capsize=2, label='Dati Filtrati')
ax3.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax3.set(title='Residui Filtrati - Discesa', xlabel='Temperatura T (K)', ylabel='Residui $\Delta P$ (Pa)')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend()

ax4.errorbar(t_sal_f, residui_sal_f, yerr=err_p_sal_f, xerr=err_t_sal_f, fmt='gx', ecolor='g', capsize=2, label='Dati Filtrati')
ax4.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax4.set(title='Residui Filtrati - Salita', xlabel='Temperatura T (K)')
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# PARTE 3: CORREZIONI SISTEMATICHE (VOLUME NON TERMALIZZATO E DILATAZIONE VETRO)
# ==============================================================================
print("\n" + "="*70 + "\n=== 3. ANALISI CORRETTA (SISTEMATICHE COMPENSATE)\n" + "="*70)

def applica_correzioni(P_mis, T):
    """Calcola la pressione ideale compensando dilatazione vetro e volume esterno."""
    fattore_vetro = 1 + gamma * (T - T_0)
    fattore_volume = (1 + (v * T) / (V0 * T_a)) / (1 + (v * T_0) / (V0 * T_a))
    return P_mis * fattore_vetro * fattore_volume

# Correzione dati filtrati
p_disc_corr = applica_correzioni(p_disc_f, t_disc_f)
p_sal_corr = applica_correzioni(p_sal_f, t_sal_f)

# Correzione incertezze sulla pressione (proporzionale al fattore di scala)
err_p_disc_corr = err_p_disc_f * (p_disc_corr / p_disc_f)
err_p_sal_corr = err_p_sal_f * (p_sal_corr / p_sal_f)

# Fit pesati sui dati corretti
mis_disc_corr = WeightedMeasurements(t_disc_f, p_disc_corr, err_t_disc_f, err_p_disc_corr)
reg_disc_corr, xy_disc_corr = weightedLinReg(mis_disc_corr)

mis_sal_corr = WeightedMeasurements(t_sal_f, p_sal_corr, err_t_sal_f, err_p_sal_corr)
reg_sal_corr, xy_sal_corr = weightedLinReg(mis_sal_corr)

# Stampa risultati finali
print(f"Chi2 ridotto discesa corretta : {reg_disc_corr.chi2/reg_disc_corr.nu:.2f}")
print(f"Chi2 ridotto salita corretta  : {reg_sal_corr.chi2/reg_sal_corr.nu:.2f}\n")

T0_disc_corr, err_T0_disc_corr = calcola_zero_assoluto(reg_disc_corr)
T0_sal_corr, err_T0_sal_corr = calcola_zero_assoluto(reg_sal_corr)

print(f"Zero assoluto (Discesa): {T0_disc_corr:.2f} +/- {err_T0_disc_corr:.2f} K  ({T0_disc_corr - T_0:.2f} °C)")
print(f"Zero assoluto (Salita) : {T0_sal_corr:.2f} +/- {err_T0_sal_corr:.2f} K  ({T0_sal_corr - T_0:.2f} °C)")
print(f"Test compatibilità 0 K : {np.abs(T0_disc_corr)/err_T0_disc_corr:.2f} (disc) | {np.abs(T0_sal_corr)/err_T0_sal_corr:.2f} (sal)\n")

# Grafico fit e Residui Corretti
print("=> Mostro i grafici del Fit Corretto")
plt.figure(30, figsize=(10, 6))
plt.title('Fit Corretto: Pressione vs Temperatura')
plt.xlabel('Temperatura T (K)')
plt.ylabel('Pressione $P$ (Pa)')
plt.grid(True, linestyle=':', alpha=0.5)
# Generazione fittizia per la legenda
plt.plot([], [], 'ro', label='Dati Corretti Discesa')
plt.plot([], [], 'k-', label='Fit Discesa')
plt.plot([], [], 'go', label='Dati Corretti Salita')
plt.plot([], [], 'b-', label='Fit Salita')
plt.legend()
plotGraphs([xy_disc_corr, mis_disc_corr, xy_sal_corr, mis_sal_corr], GraphVisuals(['k', 'ro', 'b', 'go']), figureID=30)

residui_disc_corr = p_disc_corr - (reg_disc_corr.a + reg_disc_corr.b * t_disc_f)
residui_sal_corr = p_sal_corr - (reg_sal_corr.a + reg_sal_corr.b * t_sal_f)

fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
ax5.errorbar(t_disc_f, residui_disc_corr, yerr=err_p_disc_corr, xerr=err_t_disc_f, fmt='rx', ecolor='r', capsize=2, label='Dati Corretti')
ax5.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax5.set(title='Residui Corretti - Discesa', xlabel='Temperatura T (K)', ylabel='Residui $\Delta P$ (Pa)')
ax5.grid(True, linestyle=':', alpha=0.5)
ax5.legend()

ax6.errorbar(t_sal_f, residui_sal_corr, yerr=err_p_sal_corr, xerr=err_t_sal_f, fmt='gx', ecolor='g', capsize=2, label='Dati Corretti')
ax6.axhline(0, color='k', linestyle='--', label='Residuo Atteso (=0)')
ax6.set(title='Residui Corretti - Salita', xlabel='Temperatura T (K)')
ax6.grid(True, linestyle=':', alpha=0.5)
ax6.legend()
plt.tight_layout()
plt.show()