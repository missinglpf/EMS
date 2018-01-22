# Forecasting thread for short term forecasting.
# from forecasting.short_term_forecasting import short_term_forecasting_pv, short_term_forecasting_wp, \
#     short_term_forecasting_load_ac, short_term_forecasting_load_dc, short_term_forecasting_load_uac, \
#     short_term_forecasting_load_udc

from forecasting.short_term_forecasting import short_term_forecasting_pv_history, short_term_forecasting_wp_history, \
    short_term_forecasting_load_ac_history, short_term_forecasting_load_dc_history, short_term_forecasting_load_uac_history, \
    short_term_forecasting_load_udc_history


from utils import Logger
from copy import deepcopy
logger = Logger("Short_term_forecasting")

class ForecastingThread():
    # Thread operation with time control and return value
    def short_term_forecasting(*args):
        session = args[0]
        Target_time = args[1]
        models = deepcopy(args[2])
        ACmodel = args[3]
        PVmodel = args[4]
        if models["PV"]["GEN_STATUS"] > 0:
            pv_profile = short_term_forecasting_pv_history(session, Target_time,PVmodel )
            print('....................PV profile...........................')
            print(pv_profile)
            models["PV"]["PG"] = round(models["PV"]["PMAX"] * pv_profile)
        else:
            logger.warning("No PV is connected, set to default value 0!")
            models["PV"]["PG"] = 0

        if models["WP"]["GEN_STATUS"] > 0:
            wp_profile = short_term_forecasting_wp_history(session, Target_time)
            print('....................WP profile...........................')
            print(wp_profile)
            models["WP"]["PG"] = round(models["WP"]["PMAX"] * wp_profile)
        else:
            logger.warning("No WP is connected, set to default value 0!")
            models["WP"]["PG"] = 0

        if models["Load_ac"]["STATUS"] > 0:
            load_ac = short_term_forecasting_load_ac_history(session, Target_time,ACmodel)
            models["Load_ac"]["PD"] = int(load_ac * models["Load_ac"]["PDMAX"])
            print('....................AC load profile...........................')
            print(load_ac)
        else:
            logger.warning("No critical AC load is connected, set to default value 0!")
            models["Load_ac"]["PD"] = 0

        if models["Load_uac"]["STATUS"] > 0:
            if models["Load_uac"]["STATUS"] > 0:
                load_uac = short_term_forecasting_load_uac_history(session, Target_time)
                models["Load_uac"]["PD"] = round(load_uac * models["Load_uac"]["PDMAX"])
                print('....................uAC load profile...........................')
                print(models["Load_uac"]["PD"])
            else:
                logger.warning("No non-critical AC load is connected, set to default value 0!")
                models["Load_uac"]["PD"] = 0

        if models["Load_dc"]["STATUS"] > 0:
            load_dc = short_term_forecasting_load_dc_history(session, Target_time)
            models["Load_dc"]["PD"] = round(load_dc * models["Load_dc"]["PDMAX"])
            print('....................DC load profile...........................')
            print(models["Load_dc"]["PD"])
        else:
            logger.warning("No critical DC load is connected, set to default value 0!")
            models["Load_dc"]["PD"] = 0

        if models["Load_udc"]["STATUS"] > 0:
            load_udc = short_term_forecasting_load_udc_history(session, Target_time)
            models["Load_udc"]["PD"] = round(load_udc * models["Load_udc"]["PDMAX"])
        else:
            logger.warning("No non-critical DC load is connected, set to default value 0!")
            models["Load_udc"]["PD"] = 0

        return models
