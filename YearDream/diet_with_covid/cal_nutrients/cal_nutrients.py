import pandas as pd
nutrients_M = pd.read_csv('./app/cal_nutrients/DB/nutrients_M.csv')
nutrients_W = pd.read_csv('./app/cal_nutrients/DB/nutrients_W.csv')
body_M = pd.read_csv('./app/cal_nutrients/DB/body_size_M.csv')
body_W = pd.read_csv('./app/cal_nutrients/DB/body_size_W.csv')

# 입력받은 성별, 나이, 키로 통계량 도출 [Z = (X-mean)/std] 
# // 표본수가 충분히 많아 std값으로 계산
def body_classifier(sex,age,height) :
    # 남,여로 분류 후 82세 이상, 31세 이하, 그외의 경우로 나누어 계산
    if sex == '남' :
        # 82세 이상의 경우
        if age >= 82 :
            query = body_M['연령']=='82'
            h_mean = body_M[query]['신장평균']
            h_std = body_M[query]['신장표준편차']
            # Z 값을 소수점 2자리까지만 도출
            Z = round((height-h_mean)/h_std,2)
            # Z value 만 도출
            return Z.values
        # 31세 이상의 경우
        if age <= 31 :
            query = body_M['연령']=='27~31'
            h_mean = body_M[query]['신장평균']
            h_std = body_M[query]['신장표준편차']
            # Z 값을 소수점 2자리까지만 도출
            Z = round((height-h_mean)/h_std,2)
            # Z value 만 도출
            return Z.values
        # 그 외의 경우 
        for b in body_M['연령'][2:-1] :
            if age>=int(b[:2]) and age<=int(b[3:]) :
                query = body_M['연령']==b
                h_mean = body_M[query]['신장평균']
                h_std = body_M[query]['신장표준편차']
                # Z 값을 소수점 2자리까지만 도출
                Z = round((height-h_mean)/h_std,2)
                # Z value 만 도출
                return Z.values
    # 여자의 경우도 똑같이 진행
    else :
        if age >= 82 :
            query = body_W['연령']=='82'
            h_mean = body_W[query]['신장평균']
            h_std = body_W[query]['신장표준편차']
            Z = round((height-h_mean)/h_std,2)
            return Z.values
        if age <= 31 :
            query = body_W['연령']=='27~31'
            h_mean = body_W[query]['신장평균']
            h_std = body_W[query]['신장표준편차']
            Z = round((height-h_mean)/h_std,2)
            return Z
        for b in body_W['연령'][2:-1] :
            if age>=int(b[:2]) and age<=int(b[3:]) :
                query = body_W['연령']==b
                h_mean = body_W[query]['신장평균']
                h_std = body_W[query]['신장표준편차']
                Z = round((height-h_mean)/h_std,2)
                return Z.values

# 구한 Z값을 토대로 사람별 필요한 영양소를 딕셔너리 형태로 소수점 두자리까지 도출
def nutrient(Z,sex,age,activity) :
    result = []
    if sex == '남' :
        if age >= int(nutrients_M.columns[-2][2:4]) :
            n_mean = nutrients_M[nutrients_M.columns[-2]]
            n_std = nutrients_M[nutrients_M.columns[-1]]
            for i in range(len(nutrients_M)-2):
                result.append(round(activity*(float(n_mean[i] + Z*float(n_std[i]))),2))
            return result
        for i in range(9,16,2) :
            if age >= int(nutrients_M.columns[i][:2]) and age <= float(nutrients_M.columns[i][3:5]) :
                n_mean = nutrients_M[nutrients_M.columns[i]]
                n_std = nutrients_M[nutrients_M.columns[i+1]]
                for i in range(len(nutrients_M)-2):
                    result.append(round(activity*(float(n_mean[i] + Z*float(n_std[i]))),2))
                return result
    else :
        if age >= int(nutrients_W.columns[-2][2:4]) :
            n_mean = nutrients_W[nutrients_W.columns[-2]]
            n_std = nutrients_W[nutrients_W.columns[-1]]
            for i in range(len(nutrients_M)-2):
                result.append(round(activity*(float(n_mean[i] + Z*float(n_std[i]))),2))
            return result
        for i in range(9,16,2) :
            if age >= int(nutrients_W.columns[i][:2]) and age <= float(nutrients_W.columns[i][3:5]) :
                n_mean = nutrients_W[nutrients_W.columns[i]]
                n_std = nutrients_W[nutrients_W.columns[i+1]]
                for i in range(len(nutrients_M)-2):
                    result.append(round(activity*(float(n_mean[i] + Z*float(n_std[i]))),2))
                return result