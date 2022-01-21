import pandas as pd
import numpy as np
import warnings
import os
import gc
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows",None)
pd.set_option('display.expand_frame_repr', False)


######################## FUNCTION TO ACCESS FILE ########################
def csv_file(file_name,refresh=False):
    file_path = os.getcwd()
    if file_name.find(".csv") > 0:
        dataframe_csv = pd.read_csv(file_path + '\\' + file_name)
    else:
        dataframe_csv = pd.read_csv(file_path + '\\' + file_name+ ".csv")
    print(dataframe_csv.head(),"\n")
    return dataframe_csv


def excel_file(file_name,sheet,refresh=False):
    file_path = os.getcwd()
    if file_name.find(".xlsx") > 0:
        dataframe_xlsx = pd.read_excel(file_path + '\\' + file_name,sheet_name=sheet)
    else:
        dataframe_xlsx = pd.read_excel(file_path + '\\' + file_name+ ".xlsx",sheet_name=sheet)
    print(dataframe_xlsx.head(),"\n")
    return dataframe_xlsx


def file_access(refresh=False):
    while True:
        file_choice = input("Which excel or csv file do you want to upload? If your choice is excel please enter 1, if your choice csv enter 2.")
        if file_choice not in ["1", "2"]:
            print("You did not enter a valid selection value...")
        else:
            break
    file_name = input("Please, enter the name of your file")
    files=[file_name  if  file_name  in file.name else 0 for file in os.scandir(os.getcwd())]
    dataframe = pd.DataFrame()
    if file_name in files:
        if file_choice=="1":
            try:
                sheet_name = input("Please, enter the name of sheet")
                dataframe = excel_file(file_name, sheet=sheet_name)
            except FileNotFoundError:
                print("There is no such file...")
        elif file_choice=="2":
            try:
                dataframe= csv_file(file_name)
            except FileNotFoundError:
                print("There is no such file...")
    else:
        print("There is no such file...")
    gc.collect()
    if refresh:
        dataframe.to_pickle(os.getcwd()+"/pickle_dataset/"+ file_name +".pkl")
    return dataframe
#df=file_access(refresh=True).copy()
#WORK_FILE.csv


############################## IDENTIFYING THE DISTINCTION BETWEEN WHETHER VARIABLES ARE CATEGORICAL OR NUMERICAL ##############################
def col_types(dataframe,  cat_th_ratio_upper=35,car_th_cat_th_lower=65, car_th_cat_th_upper=100):
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() / dataframe.shape[0] * 100 < cat_th_ratio_upper and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if car_th_cat_th_lower <= dataframe[col].nunique() / dataframe.shape[0] * 100 < car_th_cat_th_upper
                   and dataframe[col].dtypes == "O" and  ((r"^ *\d[\d ]*$"),col)]
    typless_cols=[col for col in dataframe.columns if col.upper() in ["NAME","SURNAME","ID"]]
    num_cols = [col for col in dataframe.columns if ((dataframe[col].dtypes != "O" and col not in num_but_cat) or col in cat_but_car) and col not in typless_cols]
    cat_cols = [col for col in dataframe.columns if ((dataframe[col].dtypes == "O" and col not in cat_but_car) or col in num_but_cat) and col not in typless_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'typless_cols: {len(typless_cols)}', "\n")
    print(f'cat_cols_name: {cat_cols}',f'num_cols_name: {num_cols}',f'cat_but_car_name: {cat_but_car}',f'typless_cols: {typless_cols}',sep="\n")

    for col in cat_but_car:
        try:
            dataframe[col].astype(float)
        except ValueError as e:
            print(col, "variable comes categorically due to the format difference. We fixed it and converted it to float !!!")
        finally:
            dataframe[col] = dataframe[col].str.replace(',', '').astype(float)
    dataframe[cat_cols]=dataframe[cat_cols].astype(str).replace('nan',np.nan)
    return cat_cols, num_cols, cat_but_car,typless_cols
#cat_cols, num_cols, cat_but_car,typless_cols =col_types(df)



######################## DESCRIPTIVE STATISTICS FOR VARIABLES IN THE DATASET ########################
def desc_statistics(dataframe,num_cols,cat_cols,head=True,count=5,shape=True,dtypes=True,
                    describe_kat=False, quantile=False,null_control=True,
                    high_null_count=True,na_rows=True,refresh=False, plot_hist=False,plot_bar=False,null_ratio=45):
    if head:
        print("DATAFRAME FIRST",count, "ROWS PREVIEW","\n",dataframe.head(count),"\n")
    if shape:
        df_row_count=dataframe.shape[0]
        df_column_count = dataframe.shape[1]
        print("ROW COUNT OF DATAFRAME    :",df_row_count)
        print("COLUMN COUNT OF DATAFRAME :",df_column_count,"\n")
    if dtypes:
        print("TYPE OF DATAFRAME :","\n",dataframe.dtypes, "\n")
    if null_control:
        null_data_count=pd.DataFrame(dataframe.isnull().sum()).reset_index().rename(columns={'index': 'Column_Name', 0: 'Value'})
        null_data_ratio = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0]).reset_index().rename(columns={'index': 'Column_Name', 0: 'Ratio'})
        null_data_ratio["Ratio"] = null_data_ratio["Ratio"].map(lambda x: "{:,.3f}".format(x))
        print("COLUMN-BASED MISSING VALUES AND RATIO:", "\n", null_data_count.merge(null_data_ratio,how="inner",on="Column_Name"), "\n")
    na_col_name=[]
    if na_rows:
        na_col_name = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        for col in dataframe[na_col_name]:
            print("Na Variable : ", col)
            print(dataframe[dataframe[col].isnull()], "\n")
    else:
        print("Since you set the na_rows parameter to false, the na_col_name field will be empty.")
    null_high_col_name={}
    if high_null_count:
        print("TOTAL NUMBER OF MISSING VALUES AND RATIO:", dataframe.isnull().sum().sum())
        print("MISSING VALUES RATIO                    :", format(dataframe.isnull().sum().sum()/dataframe.shape[0],".3f"), "\n")
        null_columns={col:dataframe[col].isnull().sum().sum()/dataframe.shape[0]*100 for col in dataframe.columns if dataframe[col].isnull().sum().sum()/dataframe.shape[0]*100>=null_ratio}.items()
        null_high_col=pd.DataFrame(null_columns, columns=["Variable", "Percent"]).sort_values(by="Percent", ascending=False)
        null_high_col_name=null_high_col["Variable"].tolist()
        if null_high_col["Variable"].nunique() == 0:
            print("THERE IS NO PROBLEM WITH THE FILL RATE IN THE DATASET...","\n")
        else:
            print("BELOW ARE THE VARIABLES WITH HIGHER THAN", str(null_ratio) + "% NULL-TO-VALUE RATIO")
            print(null_high_col,"\n")
    else:
        print("Since you set the high_null_count parameter to false, the null_high_col_name field will be empty.")
    if quantile:
        quantile = [0.05, 0.10, 0.20,0.25, 0.50, 0.75, 0.80, 0.90, 0.95]
        print("                  DESCRIPTIVE STATISTICS FOR NUMERICAL VARIABLES")
        print(dataframe[num_cols].describe(quantile).T,"\n")
        if plot_hist:
            dataframe[num_cols].hist(bins=50)
            plt.xlabel(num_cols)
            plt.title(num_cols)
            plt.show()
    if describe_kat:
        print("                  DESCRIPTIVE STATISTICS FOR CATEGORICAL VARIABLES")
        print(dataframe[cat_cols].describe().T)
        if plot_bar:
            dataframe[cat_cols].value_counts().plot(kind="bar", figsize=(10,5),color="blue")
            plt.xlabel(cat_cols)
            plt.title(cat_cols)
            plt.show()
    gc.collect()
    if refresh:
        file_path = os.getcwd()
        dataframe.to_pickle(file_path+r'\pickle_descriptive\eda.pkl')
    return na_col_name,null_high_col_name
#na_col,null_high_col_name=desc_statistics(df,num_cols,cat_cols,refresh=True)

