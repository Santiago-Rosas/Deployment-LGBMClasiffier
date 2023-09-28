from fastapi.testclient import TestClient 

from api.main import app

client= TestClient(app)

def test_null_prediction():
    response = client.post('/predict', json = {
           "sex" : 0,
            "age": 0,
           "height": 0,
            "weight": 0,
           "waistline": 0,
            "sight_left": 0,
            "sight_right":0,
           "hear_left": 0,
            "hear_right": 0,
            "SBP": 0,
            "DBP":0,
            "BLDS":0,
            "tot_chole":0,
            "HDL_chole":0,
            "triglyceride":0,
            "hemoglobin":0,
            "urine_protein":0,
            "serum_creatinine":0,
            "SGOT_AST":0,
            "SGOT_ALT":0,
           "gamma_GTP":0,
           "SMK_stat_type_cd":0

                                                    })
    assert response.status_code == 422


def test_random_prediction():
    response = client.post('/predict', json = {
            "sex" : "Male",
            "age": 36,
           "height": 175,
            "weight": 65,
           "waistline": 70,
            "sight_left": 1,
            "sight_right":1,
           "hear_left": 1,
            "hear_right": 1,
            "SBP": 120,
            "DBP":76,
            "BLDS":96,
            "tot_chole":193,
            "HDL_chole":55,
            "triglyceride":106,
            "hemoglobin":14,
            "urine_protein":1,
            "serum_creatinine":0.8,
            "SGOT_AST":23,
            "SGOT_ALT":20,
           "gamma_GTP":23,
           "SMK_stat_type_cd":1

                                                })
    assert response.status_code == 200
    assert response.json()['prediction'] == 0 or 1  


