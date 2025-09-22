# Fit-the-ARMA-model-for-any-data-set

### DEVELOPED BY: Jesubalan A
### REGISTER NO: 212223240060

# AIM:
To implement ARMA model in python.

# ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

# PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Load Excel dataset
data = pd.read_excel("Data_Train.xlsx")

# Convert journey date to datetime
data['date'] = pd.to_datetime(data['Date_of_Journey'], errors='coerce')

# Drop missing dates if any
data = data.dropna(subset=['date'])

# Group by Month and calculate average price
monthly_prices = data.groupby(data['date'].dt.to_period("M"))['Price'].mean().reset_index()
monthly_prices['date'] = monthly_prices['date'].dt.to_timestamp()
monthly_prices.rename(columns={'Price': 'avg_price'}, inplace=True)

# Extract values
X = monthly_prices['avg_price'].dropna().values
N = 1000

# Plot Month vs Price
plt.figure(figsize=(12, 6))
plt.plot(monthly_prices['date'], X, marker='o')
plt.title('Monthly Average Flight Prices')
plt.xlabel("Month")
plt.ylabel("Avg Price")
plt.grid(True)
plt.show()

# ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(1,1)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(1,1)")
plt.tight_layout()
plt.show()

# Fit ARMA(2,2)
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22, phi2_arma22 = arma22_model.arparams
theta1_arma22, theta2_arma22 = arma22_model.maparams

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_2, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(2,2)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(2,2)")
plt.tight_layout()
plt.show()

```
# OUTPUT:

## Original data:
<img width="973" height="517" alt="Screenshot 2025-09-22 153737" src="https://github.com/user-attachments/assets/d0eb2775-b9ea-4297-8117-bfae858a6cbd" />


## Autocorrelation:
<img width="970" height="236" alt="Screenshot 2025-09-22 153744" src="https://github.com/user-attachments/assets/806fe6fe-8b0e-4995-8ec6-b6e5847c1156" />


## Partial Autocorrelation:

<img width="978" height="233" alt="Screenshot 2025-09-22 153748" src="https://github.com/user-attachments/assets/05787ff4-95e0-42f8-8d6e-10c36fb2db99" />


## SIMULATED ARMA(1,1) PROCESS:

<img width="979" height="505" alt="Screenshot 2025-09-22 153755" src="https://github.com/user-attachments/assets/f3a299ee-1562-4956-ac13-1a76e9660dba" />


## Autocorrelation:

<img width="962" height="239" alt="Screenshot 2025-09-22 153801" src="https://github.com/user-attachments/assets/403a954d-36b3-4e54-9f51-ecab145b55e2" />


## Partial Autocorrelation:

<img width="976" height="241" alt="Screenshot 2025-09-22 153806" src="https://github.com/user-attachments/assets/c8a9b5c7-2660-49be-969d-2c52043d2628" />


## SIMULATED ARMA(2,2) PROCESS:

<img width="971" height="514" alt="Screenshot 2025-09-22 153813" src="https://github.com/user-attachments/assets/7ac98db1-6e7d-4291-ab4c-90687a91aeb5" />



## Autocorrelation:
<img width="969" height="244" alt="Screenshot 2025-09-22 153819" src="https://github.com/user-attachments/assets/6db5e9df-e19b-4a86-9460-a6c86f1e2a54" />


## Partial Autocorrelation:
<img width="966" height="240" alt="Screenshot 2025-09-22 153825" src="https://github.com/user-attachments/assets/2a047ab3-0720-4adc-bbbe-eda607c0a3c8" />



# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
