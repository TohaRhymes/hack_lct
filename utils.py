import pandas as pd
import os

d2_rename = {
    "Наименование": "event",
    "Источник": "src",
    "Округ": "area",
    "Дата создания во внешней системе": "date_s",
    "Дата и время завершения события во": "date_e1",
    "Дата закрытия": "date_e2",
}
d2_cols = ["unom", "event", "src", "area", "date_s", "date_e"]

d1_rename = {
    "Год постройки": "year",
    "Количество этажей": "floors",
    "Количество подъездов": "padiks",
    "Количество квартир": "apts",
    "Общая площадь": "S",
    "Общая площадь жилых помещений": "S_l",
    "Общая площадь нежилых помещений": "S_u",
    "Износ объекта (по БТИ)": "wear",
    "Материал стен": "walls",
    "Признак аварийности здания": "break",
    "Количество пассажирских лифтов": "elev",
    "Количество грузопассажирских лифтов": "elev_semiload",
    "Количество грузовых лифтов": "elev_load",
    "Очередность уборки кровли": "roof_queue",
    "Материал кровли": "roof_material",
    "UNOM": "unom",
    "Тип жилищного фонда": "found_type", 
    "Статус МКД": "status",
    "Статус управления МКД": "manage",
}

d1_cols = [
    "unom",
    "year",
    "floors",
    "padiks",
    "apts",
    "S",
    "S_l",
    "S_u",
    "wear",
    "walls",
    "break",
    "elev",
    "elev_semiload",
    "elev_load",
    "roof_queue",
    "roof_material",
    "found_type",
    "status",
    "manage",
]

rename_ru = {v: k for k, v in list(d1_rename.items())+list(d2_rename.items())}

def return_col_ciphers():
    d1_full_name = pd.read_csv('data/1_houses_full_header.csv')
    id2col = dict(d1_full_name.iloc[:1, 4:].T[0])
    col2id = {v: k for k, v in id2col.items()}
    col_ciphers = dict()
    for filename in os.listdir("data/1_col/"):
        if '~lock' in filename:
            continue
        code = filename.replace('1_', '').replace('.csv', '')
        col = id2col[f'COL_{code}']
        print(col, filename)
        col_ciphers[col] = dict(pd.read_csv(os.path.join('data/1_col', filename), skiprows=1, index_col='ID')['NAME'])
    return col_ciphers

col_ciphers = return_col_ciphers()