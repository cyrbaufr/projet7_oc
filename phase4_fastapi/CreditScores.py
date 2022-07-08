"""
This file indicates the format of features expected
to predict if a credit wll be reimbursed or not
"""

from pydantic import (StrictInt, BaseModel)

# Class which describes features used for credit scoring
class CreditScore(BaseModel):
    EXT_SOURCE_3: float
    EXT_SOURCE_2: float
    PREV_DAYS_DECISION_MIN: float
    CODE_GENDER: StrictInt
    DAYS_EMPLOYED: float
    PREV_APP_CREDIT_PERC_MIN: float
    INSTAL_DPD_MAX: float
    AMT_CREDIT: float
    DAYS_BIRTH: float
    FLAG_OWN_CAR: StrictInt
    NAME_EDUCATION_TYPE_Higher_education: StrictInt
