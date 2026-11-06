"""Tabular, SQL, and date tools for the EHRAgent MIMIC-III interface."""
from contextlib import closing
import os
from pathlib import Path
import sqlite3

import Levenshtein
import pandas as pd


def _mimic_root():
    root = Path(os.environ["EHRSQL_DATA_ROOT"]).expanduser()
    mimic_root = root / "mimic_iii"
    if not mimic_root.is_dir():
        raise FileNotFoundError(f"MIMIC-III data directory not found: {mimic_root}")
    return mimic_root

TABLE_FILES = {
        "admissions": "ADMISSIONS.csv", "chartevents": "CHARTEVENTS.csv",
        "cost": "COST.csv", "d_icd_diagnoses": "D_ICD_DIAGNOSES.csv",
        "d_icd_procedures": "D_ICD_PROCEDURES.csv", "d_items": "D_ITEMS.csv",
        "d_labitems": "D_LABITEMS.csv", "diagnoses_icd": "DIAGNOSES_ICD.csv",
        "icustays": "ICUSTAYS.csv", "inputevents_cv": "INPUTEVENTS_CV.csv",
        "labevents": "LABEVENTS.csv", "microbiologyevents": "MICROBIOLOGYEVENTS.csv",
        "outputevents": "OUTPUTEVENTS.csv", "patients": "PATIENTS.csv",
        "prescriptions": "PRESCRIPTIONS.csv", "procedures_icd": "PROCEDURES_ICD.csv",
        "transfers": "TRANSFERS.csv",
    }


def db_loader(target_ehr):
    return pd.read_csv(_mimic_root() / TABLE_FILES[target_ehr])

def data_filter(data, argument):
    backup_data = data
    commands = argument.split('||')
    for i in range(len(commands)):
        try:
            if '>=' in commands[i]:
                command = commands[i].split('>=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] >= value]
            elif '<=' in commands[i]:
                command = commands[i].split('<=')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] <= value]
            elif '>' in commands[i]:
                command = commands[i].split('>')
                column_name = command[0]
                value = command[1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] > value]
            elif '<' in commands[i]:
                command = commands[i].split('<')
                column_name = command[0]
                value = command[1]
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    value = type(data[column_name][0])(value)
                except:
                    value = value
                data = data[data[column_name] < value]
            elif '=' in commands[i]:
                command = commands[i].split('=')
                column_name = command[0]
                value = command[1]
                if value[0] == "'" or value[0] == '"':
                    value = value[1:-1]
                try:
                    examplar = backup_data[column_name].tolist()[0]
                    value = type(examplar)(value)
                except:
                    value = value
                data = data[data[column_name] == value]
            elif ' in ' in commands[i]:
                command = commands[i].split(' in ')
                column_name = command[0]
                value = command[1]
                value_list = [s.strip() for s in value.strip("[]").split(',')]
                value_list = [s.strip("'").strip('"') for s in value_list]
                value_list = list(map(type(data[column_name][0]), value_list))
                data = data[data[column_name].isin(value_list)]
            elif 'max' in commands[i]:
                command = commands[i].split('max(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].max()]
            elif 'min' in commands[i]:
                command = commands[i].split('min(')
                column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].min()]
        except:
            if column_name not in data.columns.tolist():
                columns = ', '.join(data.columns.tolist())
                raise Exception("The filtering query {} is incorrect. Please modify the column name or use LoadDB to read another table. The column names in the current DB are {}.".format(commands[i], columns))
            if column_name == '' or value == '':
                raise Exception("The filtering query {} is incorrect. There is syntax error in the command. Please modify the condition or use LoadDB to read another table.".format(commands[i]))
        if len(data) == 0:
            # Suggest up to five nearby values when an equality filter misses.
            column_values = list(set(backup_data[column_name].tolist()))
            if ('=' in commands[i]) and (not value in column_values) and (not '>=' in commands[i]) and (not '<=' in commands[i]):
                levenshtein_dist = {}
                for cv in column_values:
                    levenshtein_dist[cv] = Levenshtein.distance(str(cv), str(value))
                levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
                column_values = [i[0] for i in levenshtein_dist[:5]]
                column_values = ', '.join([str(i) for i in column_values])
                raise Exception("The filtering query {} is incorrect. There is no {} value in the column. Five example values in the column are {}. Please check if you get the correct {} value.".format(commands[i], value, column_values, column_name))
            else:
                return data
    return data

def get_value(data, argument):
    try:
        commands = argument.split(', ')
        if len(commands) == 1:
            column = argument
            while column[0] == '[' or column[0] == "'":
                column = column[1:]
            while column[-1] == ']' or column[-1] == "'":
                column = column[:-1]
            if len(data) == 1:
                return str(data.iloc[0][column])
            else:
                answer_list = list(set(data[column].tolist()))
                answer_list = [str(i) for i in answer_list]
                return ', '.join(answer_list)
        else:
            column = commands[0]
            if 'mean' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)/len(res_list)
            elif 'max' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return max(res_list)
            elif 'min' in commands[-1]:
                res_list = data[column].tolist()
                try:
                    res_list = [float(i) for i in res_list]
                except:
                    res_list = [str(i) for i in res_list]
                return min(res_list)
            elif 'sum' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [float(i) for i in res_list]
                return sum(res_list)
            elif 'list' in commands[-1]:
                res_list = data[column].tolist()
                res_list = [str(i) for i in res_list]
                return list(res_list)
            else:
                raise Exception("The operation {} contains syntax errors. Please check the arguments.".format(commands[-1]))
    except:
        column_values = ', '.join(data.columns.tolist())
        raise Exception("The column name {} is incorrect. Please check the column name and make necessary changes. The columns in this table include {}.".format(column, column_values))

def sql_interpreter(command):
    uri = (_mimic_root() / "mimic_iii.db").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute(command).fetchall()

def date_calculator(argument):
    try:
        uri = (_mimic_root() / "mimic_iii.db").resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            results = connection.execute("select datetime(current_time, ?)", (argument,)).fetchone()[0]
    except:
        raise Exception("The date calculator {} is incorrect. Please check the syntax and make necessary changes. For the current date and time, please call Calendar('0 year').".format(argument))
    return results
