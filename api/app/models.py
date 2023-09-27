from pydantic import BaseModel, Field

# Input for data validation
class Input(BaseModel):
    sex : str =  Field(min_length=4, max_length=6)
    age: int = Field(gt=0)
    height: int = Field(gt=0)
    weight: int = Field(gt=0)
    waistline: float = Field(ge=0)
    sight_left: float = Field(ge=0)
    sight_right: float = Field(ge=0)
    hear_left: float = Field(ge=1,le=2)
    hear_right: float = Field(ge=1,le=2)
    SBP: float = Field(gt=0)
    DBP:float = Field(gt=0)
    BLDS:float = Field(gt=0)
    tot_chole:float = Field(gt=0)
    HDL_chole:float = Field(ge=1)
    triglyceride:float = Field(ge=1)
    hemoglobin:float = Field(ge=1)
    urine_protein:float = Field(ge=1)
    serum_creatinine:float = Field(ge=0)
    SGOT_AST:float = Field(ge=1)
    SGOT_ALT:float = Field(ge=1)
    gamma_GTP:float = Field(ge=1)
    SMK_stat_type_cd:float = Field(ge=1)
    
    model_config = {
        "json_schema_extra" : {
            "examples":[
                {
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

            }
            
        ]
 }
}
    
    


# Ouput 
class Output(BaseModel):
    label: str
    prediction: int


 
