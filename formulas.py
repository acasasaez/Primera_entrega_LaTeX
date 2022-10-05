import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

base_dir = "https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True"

data = pd.read_csv(base_dir)

# Convert the datetime from str to datetime object.
data["datetime"] = pd.to_datetime(data["datetime"])

# Determine the daily change in gold price.
data["gold_price_change"] = data["gold_price_usd"].diff()

# Restrict the data to later than 2008 Jan 01.
data = data[data["datetime"] >= pd.to_datetime("2008-01-01")]

# Plot the daily gold prices as well as the daily change.
plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
plt.plot(data["datetime"], data["gold_price_usd"])
plt.xlabel("datetime")
plt.ylabel("gold price (usd)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(data["datetime"], data["gold_price_change"])
plt.xlabel("datetime")
plt.ylabel("gold price change (usd)")
plt.grid(True)
plt.show()
data = pd.read_csv(base_dir)

# Convert the datetime from str to datetime object.
data["datetime"] = pd.to_datetime(data["datetime"])

# Determine the daily change in gold price.
data["gold_price_change"] = data["gold_price_usd"].diff()

# Restrict the data to later than 2008 Jan 01.
data = data[data["datetime"] >= pd.to_datetime("2008-01-01")]

plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
plt.plot(data["datetime"], data["gold_price_usd"])
plt.xlabel("datetime")
plt.ylabel("gold price (usd)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(data["datetime"], data["gold_price_change"])
plt.xlabel("datetime")
plt.ylabel("gold price change (usd)")
plt.grid(True)
plt.show()



























1.\begin{equation}
\mathcal{S} =  \left\lbrace \ s_i,\ i=1,\cdots , \ N_s \right\rbrace
\end{equation}

2.  \begin{equation} Y_t =  \left\lbrace \ y_t^i,\ i=1,\cdots , \ N_s \right\rbrace
\end{equation}

3.  \begin{equation}Y =  \left\lbrace \ Y_t, \ t=1,\cdots , \ T_m \ \right\rbrace = \ \left\lbrace \ T^i,\ i=1,\ \cdots , \ N_m \right\rbrace\end{equation}

4. \begin{equation} \mathcal{R}=  \left\lbrace \ R_i,\ i=1,\cdots ,\ K \right\rbrace\end{equation}

5. \begin{equation}d: \mathcal{F} \times \mathcal{F} \longrightarrow \mathbb{R}^+ , X^i,r_j^i \in \mathcal{F} \end{equation}
6. \begin{equation}\mathcal{C}\ \in\  \mathbb{R}^{N_x\times N_m}  \ \ \mathcal{C}\left(\ n,\ m\right) = d\left({X}^n , {R_j}^m\right)\end{equation} 

7.\begin{equation}d\left({X}^n , {R_j}^m\right)= \sqrt{\sum_{ \i = 1}^{N_s}\left(x_i^n-r_i^m^j\right)^2 }  \end{equation}

16.  \begin{equation}P\left(\mathcal{O} |\lambda \right) = 
\sum_v P\left(\mathcal{O} |V,\lambda \right)P\left(V |\lambda \right) = \sum_{v}\prod_{i = 1}^{z_o}   P\left(o_i\ |\ v_i,\lambda \right)
\left(\pi_v_1\centerdot a_v_1_v_2\centerdot... \centerdot a_v__z_v-_1 v_z_v\right)  = \sum_{v1,v2,v_z_v}\pi_v_1\ \centerdot\ b_v_1\left(o_1\right)   \centerdot\ a_v_1_v_2 \centerdot \ b_v_2\left(o_2\right)\ \centerdot... \ \centerdot\ a_v__z_v-_1 v_z_v\ \centerdot \ b_v_z_v \left(o_z_o\right)\end{equation}