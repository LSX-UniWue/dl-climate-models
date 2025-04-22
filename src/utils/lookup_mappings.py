ERA_TO_CMIP_VAR_NAME = {
    't2m': 'tas',
    'u10': 'uas',
    'v10': 'vas',
    't':'ta',
    'q':'hus',
    'z':'zg',
    'u':'ua',
    'v':'va'
}

CMIP_TO_ERA_VAR_NAME = {v: k for k, v in ERA_TO_CMIP_VAR_NAME.items()}

VAR_TO_CMIP_NAME = {
    **ERA_TO_CMIP_VAR_NAME,
}

VAR_NAMES_TO_UNIT = {
        'tas': 'K',
        'uas': 'm/s',
        'vas': 'm/s',
        'zg': 'm',
        'va': 'm/s',
        'ua': 'm/s',
        'ta': 'K',
        'hur': '%',
        'hus': 'kg/kg',
        'mole_fraction_of_carbon_dioxide_in_air':'ppm',
        'tos':'K',
        'siconc':'%',      
}