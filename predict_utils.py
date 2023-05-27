final_cont_features = [
 'year',
 'floors',
 'padiks',
 'apts',
 'S',
 'S_l',
 'S_u',
 'elev',
 'elev_semiload',
 'elev_load',]
final_cat_features = [
 'wear',
 'walls',
 'break',
 'roof_queue',
 'roof_material',
 'found_type',
 'status',
 'manage']
final_features = final_cont_features+final_cat_features

incident_features = ['Перекрытие_between',
 'Лифт_between',
 'ПУ_between',
 'Вода_between',
 'Мусор_between',
 'Свет_between',
 'Электричество_between',
 'Канализация_between',
 'Асфальт_between',
 'Внешние признаки_between',
 'Почта_between',
 'Отопление_between',
 'Обустройство_between',
 'Подтопление_between',
 'Колодец_between',
 'Ремонт_between',
 'Авария_between',
 'УСПД_between',
 'Значения_between',
 'Пожар_between',
 'Дверь_between',
 'Особое_between',
 'ГВС_between',
 'Животные_between',
 'Дорога_between',
 'Газ_between',
 'Мусоропровод_between',
 'Трубопровод_between',
 'Окна_between',
 'Связь_between',
 'Крыша_between']


capital_features = ['ХВС_c_between',
 'ВДСК_c_between',
 'Подъезд_c_between',
 'ГВС_c_between',
 'Мусоропровод_c_between',
 'КАН-М_c_between',
 'ГАЗ_c_between',
 'ЭС_c_between',
 'Крыша_c_between',
 'ГВС-М_c_between',
 'КАН_c_between',
 'Фасад_c_between',
 'Подвал_c_between',
 'ПВ_c_between',
 'Лифт_c_between',
 'ЦО_c_between',
 'ЦО-М_c_between',
 'Окна_c_between',
 'ХВС-М_c_between']

event_to_sys = {'Авария': 'MGI',
 'Асфальт': 'MGI',
 'Внешние признаки': 'NG',
 'Вода': 'MOS_GAS',
 'ГВС': 'NG',
 'Газ': 'ASUPR',
 'Дверь': 'NG',
 'Дорога': 'MOS_GAS',
 'Животные': 'EDC',
 'Значения': 'ASUPR',
 'Канализация': 'NG',
 'Колодец': 'MOS_GAS',
 'Крыша': 'MGI',
 'Лифт': 'MGI',
 'Мусор': 'NG',
 'Мусоропровод': 'MGI',
 'Обустройство': 'MVK',
 'Окна': 'MGI',
 'Особое': 'MVK',
 'Отопление': 'MGI',
 'ПУ': 'ASUPR',
 'Перекрытие': 'NG',
 'Подтопление': 'MVK',
 'Пожар': 'NG',
 'Почта': 'NG',
 'Ремонт': 'EDC',
 'Свет': 'NG',
 'Связь': 'ASUPR',
 'Трубопровод': 'NG',
 'УСПД': 'ASUPR',
 'Электричество': 'ASUPR'}