if st.button("Evaluar Riesgo de Cliente", type="primary"):
    try:
        input_dict = {
            'region': region, 'edad': float(edad), 'genero': genero, 'tipo_contrato': tipo_contrato,
            'antiguedad_meses': float(antiguedad_meses), 'plan': plan, 'tiene_internet': float(tiene_internet),
            'velocidad_mbps': float(velocidad_mbps), 'tiene_tv': float(tiene_tv), 'tiene_linea_movil': float(tiene_linea_movil),
            'num_servicios': float(num_servicios), 'factura_mensual_clp': float(factura_mensual_clp),
            'metodo_pago': metodo_pago, 'dias_mora_hist': float(dias_mora_hist), 'reclamos_12m': float(reclamos_12m),
            'llamadas_soporte_6m': float(llamadas_soporte_6m), 'nps': float(nps), 'descuento_activo': descuento_activo,
            'meses_sin_reajuste': float(meses_sin_reajuste), 'ingreso_estimado_clp': float(ingreso_estimado_clp),
            'cambios_plan_12m': float(cambios_plan_12m)
        }
        X_input = pd.DataFrame([input_dict])

        num_cols_originales = imputer_num.feature_names_in_.tolist()
        cat_cols_originales = imputer_cat.feature_names_in_.tolist()

        X_num_imp = imputer_num.transform(X_input[num_cols_originales])
        X_cat_imp = imputer_cat.transform(X_input[cat_cols_originales])
        
        # 1. Transformar categorías
        X_cat_enc = encoder.transform(X_cat_imp)

        # 2. Calcular variables derivadas numéricas
        ratio_factura_ingreso = float(factura_mensual_clp) / (float(ingreso_estimado_clp) + 1.0)
        indice_conflictividad = float(reclamos_12m) + float(llamadas_soporte_6m)
        
        derivadas = np.array([[ratio_factura_ingreso, indice_conflictividad]])
        X_num_total = np.hstack((X_num_imp, derivadas))
        
        # 3. Obtener nombres de columnas para reconstruir el set completo transitorio
        # (Esto nos sirve para identificar el desajuste)
        num_names = num_cols_originales + ['ratio_factura_ingreso', 'indice_conflictividad']
        cat_names = encoder.get_feature_names_out(cat_cols_originales).tolist()
        all_features_names = num_names + cat_names
        
        # Juntamos los datos en un DataFrame temporal
        features_combined = np.hstack((X_num_total, X_cat_enc))
        df_temporal = pd.DataFrame(features_combined, columns=all_features_names)

        # =========================================================================
        # 🔥 EL PASO CORRECTOR: ALINEACIÓN FORZADA CON EL MODELO ENTRENADO 🔥
        # =========================================================================
        if hasattr(model, 'feature_names_in_'):
            # Si tu RandomForest se guardó conociendo el nombre de sus columnas (ideal):
            columnas_entrenamiento = model.feature_names_in_
            df_final = df_temporal.reindex(columns=columnas_entrenamiento, fill_value=0)
            features_procesadas = df_final.values
        else:
            # Si se guardó como un array de Numpy puro sin nombres, forzamos el recorte/relleno
            # a las 55 características exactas que tu modelo necesita.
            num_caracteristicas_esperadas = 55
            features_procesadas = df_temporal.values
            if features_procesadas.shape[1] > num_caracteristicas_esperadas:
                features_procesadas = features_procesadas[:, :num_caracteristicas_esperadas]
            elif features_procesadas.shape[1] < num_caracteristicas_esperadas:
                relleno = np.zeros((features_procesadas.shape[0], num_caracteristicas_esperadas - features_procesadas.shape[1]))
                features_procesadas = np.hstack((features_procesadas, relleno))
        # =========================================================================

        # 4. Realizar la predicción de manera segura con el shape correcto (1, 55)
        probabilidad = model.predict_proba(features_procesadas)[0][1]
        prob_porcentaje = probabilidad * 100

        st.write("---")
        st.subheader("📈 Resultado del Análisis de Riesgo")
        
        if probabilidad >= 0.60:
            st.error(f"⚠️ **Riesgo CRÍTICO / ALTO:** El cliente presenta un **{prob_porcentaje:.2f}%** de probabilidad de caer en mora severa.")
            st.markdown("**Acción recomendada:** Gatillar campaña de cobranza preventiva inmediata.")
        elif probabilidad >= 0.30:
            st.warning(f"🟡 **Riesgo MEDIO:** El cliente presenta un **{prob_porcentaje:.2f}%** de probabilidad de retraso.")
            st.markdown("**Acción recomendada:** Monitorear comportamiento de pago.")
        else:
            st.success(f"🟢 **Riesgo BAJO:** El cliente se mantiene estable con un **{prob_porcentaje:.2f}%** de probabilidad de mora.")
            
    except Exception as error_ejecucion:
        st.error(f"🚨 Error en procesamiento: {error_ejecucion}")
