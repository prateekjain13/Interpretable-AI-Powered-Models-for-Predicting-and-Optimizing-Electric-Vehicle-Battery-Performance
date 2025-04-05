The increasing reliance on battery-powered
systems in electric vehicles (EVs), IoT devices,
and industrial applications necessitates accurate
prediction of battery health and Remaining Useful
Life (RUL). Unexpected battery failures can lead
to safety risks, financial losses, and operational
inefficiencies. Predictive maintenance, enabled by
data-driven machine learning (ML) models, offers
a solution by forecasting degradation patterns and
enabling proactive interventions.
This research presents a robust ML-based
framework for battery RUL prediction and health
classification. The approach integrates multiple
regression models, including Extra Trees
Regressor, LightGBM, XGBoost, and Bagging
Regressor, to estimate RUL. Based on predicted
RUL, a Random Forest Classifier categorizes
battery health into three states: Healthy, Moderate
Wear, and Critical Condition. The models are
trained on the NASA Battery Prognostics dataset,
which contains charge-discharge cycle data with
features such as voltage, current, temperature, and
capacity. Feature engineering and preprocessing
techniques, including missing value imputation
and scaling, enhance the model’s reliability.
To ensure transparency, SHAP (SHapley
Additive exPlanations) analysis is used for feature
importance evaluation, highlighting key
parameters influencing battery degradation. The
deployment pipeline integrates a real-time alerting
system using SendGrid for email notifications and
Twilio for SMS alerts, ensuring early warnings in
critical conditions.
The proposed ML system demonstrates
high predictive accuracy, strong generalization
through cross-validation, and effective real-world
applicability. By combining predictive modeling
with explainability and real-time alerting, this
research aims to enhance battery maintenance
strategies, minimize downtime, and improve the
reliability of battery-powered systems.
