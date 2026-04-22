import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GFS_COARSE_SELECTED = {
    "surface_wind_10m": ["UGRD_588", "VGRD_589", "GUST_14"],
    "surface_wind_100m": ["UGRD_689", "VGRD_690"],
    "surface_temp_2m": ["TMP_581", "DPT_583", "RH_584", "TMAX_586", "TMIN_587"],
    "surface_pressure": ["PRMSL_1", "PRES_639"],
    "solar_radiation": ["DSWRF_653", "DLWRF_654", "SUNSD_622"],
    "cloud": ["LCDC_630", "MCDC_632", "HCDC_634", "TCDC_636"],
    "precipitation": ["PRATE_593", "APCP_596", "CRAIN_604", "CSNOW_601"],
    "flux": ["LHTFL_609", "SHTFL_610"],
    "convective": ["CAPE_624", "CIN_625", "HPBL_712", "PWAT_626"],
    "isobaric_1000mb": ["TMP_196", "RH_197", "UGRD_202", "VGRD_203"],
    "isobaric_925mb": ["TMP_286", "RH_287", "UGRD_292", "VGRD_293"],
    "isobaric_850mb": ["TMP_302", "RH_303", "TCDC_304", "UGRD_308", "VGRD_309"],
    "isobaric_700mb": ["TMP_350", "RH_351", "TCDC_352", "UGRD_356", "VGRD_357"],
    "isobaric_500mb": ["TMP_446", "RH_447", "UGRD_452", "VGRD_453"],
    "isobaric_300mb": ["TMP_494", "UGRD_500", "VGRD_501"],
    "soil_top": ["TSOIL_564", "SOILW_565"],
    "surface_extra": ["ALBDO_730", "WEASD_577", "SNOD_578"],
}

META_COLUMNS = ["TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE", "LON", "LAT"]


def get_download_columns() -> list:
    cols = list(META_COLUMNS)
    for group_cols in GFS_COARSE_SELECTED.values():
        cols.extend(group_cols)
    seen = set()
    unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


WIND_RAW = {
    "10m": ("UGRD_588", "VGRD_589"),
    "100m": ("UGRD_689", "VGRD_690"),
    "1000mb": ("UGRD_202", "VGRD_203"),
    "925mb": ("UGRD_292", "VGRD_293"),
    "850mb": ("UGRD_308", "VGRD_309"),
    "700mb": ("UGRD_356", "VGRD_357"),
    "500mb": ("UGRD_452", "VGRD_453"),
    "300mb": ("UGRD_500", "VGRD_501"),
}

SOLAR_RAW = {
    "dswrf": "DSWRF_653",
    "dlwrf": "DLWRF_654",
    "sunsd": "SUNSD_622",
    "tcdc_low": "LCDC_630",
    "tcdc_mid": "MCDC_632",
    "tcdc_high": "HCDC_634",
    "tcdc_total": "TCDC_636",
}

TEMP_RAW = {
    "tmp_2m": "TMP_581",
    "dpt_2m": "DPT_583",
    "rh_2m": "RH_584",
    "tmax": "TMAX_586",
    "tmin": "TMIN_587",
}

PLEVEL_TEMP = {
    "1000mb": "TMP_196",
    "925mb": "TMP_286",
    "850mb": "TMP_302",
    "700mb": "TMP_350",
    "500mb": "TMP_446",
    "300mb": "TMP_494",
}

PLEVEL_RH = {
    "1000mb": "RH_197",
    "925mb": "RH_287",
    "850mb": "RH_303",
    "700mb": "RH_351",
    "500mb": "RH_447",
}
