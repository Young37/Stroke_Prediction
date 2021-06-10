import pandas as pd # 판다스 라이브러리 호출
#행과 열로 이루어진 DataFrame Type데이터를 조작할 때 편한 라이브러리 => pandas

# pandas의 read_csv()함수로 외부 csv파일을 불러와서 DataFrame으로 저장합니다.
csv_test = pd.read_csv('C:/Users/skvsn/Desktop/stroke/healthcare-dataset-stroke-data.csv',index_col='id')

# 행과 열의 개수 파악합니다
print(csv_test.shape)

# 데이터 전체 출력합니다.
print(csv_test)

# 데이터 타입을 출력합니다.
print(csv_test.dtypes)

# 숫자 타입으로 변환합니다.
csv_test['gender'] = pd.Categorical(csv_test['gender'])
csv_test['gender'] = csv_test.gender.cat.codes

csv_test['ever_married'] = pd.Categorical(csv_test['ever_married'])
csv_test['ever_married'] = csv_test.ever_married.cat.codes

csv_test['work_type'] = pd.Categorical(csv_test['work_type'])
csv_test['work_type'] = csv_test.work_type.cat.codes

csv_test['Residence_type'] = pd.Categorical(csv_test['Residence_type'])
csv_test['Residence_type'] = csv_test.Residence_type.cat.codes

csv_test['smoking_status'] = pd.Categorical(csv_test['smoking_status'])
csv_test['smoking_status'] = csv_test.smoking_status.cat.codes

# 데이터 전체 출력합니다.
print(csv_test)

# 데이터 타입을 출력합니다.
print(csv_test.dtypes)