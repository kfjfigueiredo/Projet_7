#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

# Page setting
st.set_page_config(page_title='Dashboard Scoring Credit - Prêt à dépenser ',  layout='wide')

# Telecharger la base de données
df = pd.read_csv('https://raw.githubusercontent.com/kfjfigueiredo/Projet_7/main/df_train_f2.csv',index_col= "SK_ID_CURR")
img =Image.open('Logo_pad.PNG')
graphique_shap_importance = Image.open('Shap_importance.png')


#header
t1, t2 = st.columns((0.07,1)) 

t1.image(img, width = 120) #logo
t2.title("Dashboard Scoring Credit ") # Titre du dashboard 

# Filtre pour choisir le client: 
df['id'] = df['id'].astype(str) # transformer l'ID en string 
id_client = st.selectbox('Selectionnez un Id client', df['id'], help = 'Choisissez un seul id client')

# Type de client:
classe_predit = df[(df['id']==id_client) & (df['type_de_client'])]
classe_predit = classe_predit[["type_de_client"]]

# Classe réele:
df['classe_reel'] = "p"
df['classe_reel'] = np.where((df['target_reel']== 1), "avec défault", df['classe_reel'])
df['classe_reel'] = np.where((df['target_reel']== 0), "sans défault", df['classe_reel'])
classe_reele = df[(df['id']==id_client) & (df['classe_reel'])]


# probabilité de deffaillance:
df['prob_defaut'] = df['probabilite_default']*100
prob = df[(df['id']==id_client) & (df['prob_defaut'])]
prob = prob[['prob_defaut']]

# probabilité de payement:
#df['prob_pay'] = 1 - df['probabilite_defaut']
#df['prob_pay'] = df['prob_pay'] *100 
#prob_pay = df[(df['id']==id_client) & (df['prob_pay'])]
    

#affichage de la prédiction
chaine = '**Type de client :**' + str(classe_predit) 
st.markdown(chaine)

chaine2 = '**Risque de défault :**' + str(prob) + '% de risque de défaut'
st.markdown(chaine2)


# PARTIE GRAPHIQUE 

g1, g2, g3 = st.columns((1,1,1))


# 1er graph:
g1.subheader("Ranking des features importances avec SHAP ") # Titre du dashboard 
g1.image(graphique_shap_importance, width = 500) #graphique de features importance avec Shap


# 2eme graph:

import plotly.express as px
import plotly.graph_objects as go

fig1 = px.box(df, x= "type_de_client", y= "PAYMENT_RATE", title= f'Payment Rate par Target')
fig1.update_traces(marker_color='#264653')
fig2 =  px.scatter(x = df['type_de_client'][(df["id"] == id_client)], y = df['PAYMENT_RATE'][(df["id"] == id_client)])
fig2.update_traces(marker_color= 'red')
fig3 = go.Figure(data=fig1.data + fig2.data)
#fig3.update_layout(title_text="Payement Rate par rapport au type de client ",title_x=0,margin= dict(l=5,r=5,b=10,t=30), yaxis_title=None, xaxis_title=None)
g2.subheader("PAYMENT RATE x CLIENT TYPE")
g2.plotly_chart(fig3, use_container_width=False)

# 3eme graph  

fig4 = px.box(df, x= "CODE_GENDER", y= "DAYS_BIRTH")
fig4.update_traces(marker_color='#264653')
fig5 =  px.scatter(x = df['CODE_GENDER'][(df["id"] == id_client)], y = df['DAYS_BIRTH'][(df["id"] == id_client)])
fig5.update_traces(marker_color= 'red')
fig6 = go.Figure(data=fig4.data + fig5.data)
g3.subheader("DAYS BIRTH x GENDER_CODE")
g3.plotly_chart(fig6, use_container_width=True)


# 2ème ligne de graphs:

l1, l2, l3 = st.columns((1,1,1))

#4ème graph : 
fig7 = px.box(df, x= "NAME_FAMILY_STATUS_Married", y= "EXT_SOURCE_2")
fig7.update_traces(marker_color='#264653')
fig8 =  px.scatter(x = df['NAME_FAMILY_STATUS_Married'][(df["id"] == id_client)], y = df['EXT_SOURCE_2'][(df["id"] == id_client)])
fig8.update_traces(marker_color= 'red')
fig9 = go.Figure(data=fig7.data + fig8.data)
l1.subheader("EXT_SOURCE_2 x Marié (0= non/ 1= oui)")
l1.plotly_chart(fig9, use_container_width=True)

#5ème graph : 
fig10 = px.box(df, x= "NAME_EDUCATION_TYPE", y= "EXT_SOURCE_2")
fig10.update_traces(marker_color='#264653')
fig11 =  px.scatter(x = df['NAME_EDUCATION_TYPE'][(df["id"] == id_client)], y = df['EXT_SOURCE_2'][(df["id"] == id_client)])
fig11.update_traces(marker_color= 'red')
fig12 = go.Figure(data=fig10.data + fig11.data)
l2.subheader("EXT_SOURCE_2 x Education Type")
l2.plotly_chart(fig12, use_container_width=True)

#6ème graph : 
fig10 = px.box(df, x= "NAME_EDUCATION_TYPE", y= "EXT_SOURCE_3")
fig10.update_traces(marker_color='#264653')
fig11 =  px.scatter(x = df['NAME_EDUCATION_TYPE'][(df["id"] == id_client)], y = df['EXT_SOURCE_3'][(df["id"] == id_client)])
fig11.update_traces(marker_color= 'red')
fig12 = go.Figure(data=fig10.data + fig11.data)
l3.subheader("EXT_SOURCE_3 x Education Type")
l3.plotly_chart(fig12, use_container_width=True)


#ajout informations concernant variables:

k1, k2, k3 = st.columns((1,1,1))
k1.subheader("Explication des variables")

'''
st.markdown("\
\n\
* **Client** : la valeur pour le client considéré\n\
* Moyenne : valeur moyenne pour l'ensemble des clients\n\
* En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
* En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
* Similaires : valeur moyenne pour les 20 clients les plus proches du client\
considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
")
'''


#Run sur streamlit: streamlit run "C:/Users/kathl/Desktop/Projet7_OP/streamlit_app.py

