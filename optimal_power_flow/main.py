# Main entrance for the short-term operation of both universal ems and local ems
import threading
import time
from configuration.configuration_time_line import default_time

from optimal_power_flow.short_term_forecasting import ForecastingThread
from configuration.configuration_time_line import default_dead_line_time
from optimal_power_flow.set_ponits_tracing import set_points_tracing_opf

from utils import Logger
from configuration.configuration_time_line import default_look_ahead_time_step
from copy import deepcopy
from optimal_power_flow.input_check import input_check_short_term
from optimal_power_flow.output_check import output_local_check

logger_uems = Logger("Short_term_dispatch_UEMS")
logger_lems = Logger("Short_term_dispatch_LEMS")

class short_term_operation():
    ##short term operation for ems

    def short_term_operation_lems(*args):
        from data_management.database_management import database_operation
        from optimal_power_flow.problem_formulation import problem_formulation
        from optimal_power_flow.problem_solving import Solving_Thread
        # Short term operation for local ems
        # The following operation sequence
        # 1) Information collection
        # 2) Short-term forecasting
        # 3) Information upload and database store
        # 4) Download command and database operation
        local_models = deepcopy(args[0])  # Local energy management system models
        session = args[1]  # local database
        ACmodel = args[2]
        PVmodel = args[3]
        Target_time = time.time()
        print('....................Target time...........................')
        print(Target_time)
        Target_time = round((Target_time - Target_time % default_time["Time_step_opf"] + default_time[
            "Time_step_opf"]))
        print('....................Target time opf...........................')
        print(Target_time)
        # Step 1: Short-term forecasting
        thread_forecasting = ForecastingThread.short_term_forecasting(session, Target_time, local_models, ACmodel,PVmodel)  # The forecasting thread
        local_models = thread_forecasting
        # Solve the optimal power flow problem
        local_models = input_check_short_term.model_local_check(local_models)

        #logger_uems.info("OPF is under idle mode!")
        mathematical_model = problem_formulation.problem_formulation_universal(local_models, "Feasible")
        #mathematical_model_recovery = problem_formulation.problem_formulation_universal(local_models, universal_models,
           #                                                                             "Infeasible")
        local_models["COMMAND_TYPE"] = 0


        # Solving procedure
        res = Solving_Thread.solving_procedure(mathematical_model)
        #res_recovery = Solving_Thread(mathematical_model_recovery)
        #res.daemon = True
        #res_recovery.daemon = True

        local_models = result_update(res, local_models,"Feasible")


        # The output check the result
        local_models = output_local_check(local_models)
        # Update the dynamic model
       # local_models = input_check_short_term.model_local_check(local_models) # Check the data format of local ems
       # local_models = set_points_tracing_opf(Target_time,session,local_models) # Update the


        # Information send

        # Step2: Backup operation, which indicates the universal ems is down


        # Store the data into the database


        #Check the output of optimal power flow
        # local_models = output_local_check(local_models)

        database_operation.database_record(session, local_models, Target_time, "OPF")
        print('.............results.................')
        print(Target_time)
        print(local_models)

def result_update(*args):
    ## Result update for local ems and universal ems models
    res = args[0]
    local_model = args[1]
   # universal_model = args[2]
    type = args[2]

    if type == "Feasible":
        if local_model["COMMAND_TYPE"] is 0:
            from modelling.power_flow.idx_format import NX
        else:
            from modelling.power_flow.idx_opf_set_points_tracing import NX
    else:
        if local_model["COMMAND_TYPE"] is 0:
            from modelling.power_flow.idx_format_recovery import NX
        else:
            from modelling.power_flow.idx_opf_set_points_tracing_recovery import NX

    x_local = res["x"][0:NX]
   # x_universal = res["x"][NX:2 * NX]

    local_model = update(x_local, local_model, type)
   # universal_model = update(x_universal, universal_model, type)

    return local_model


def update(*args):
    x = args[0]
    model = args[1]
    model_type = args[2]

    if model_type == "Feasible":
        if model["COMMAND_TYPE"] is 0:
            from modelling.power_flow.idx_format import PG, QG, RG, PUG, QUG, RUG, PBIC_AC2DC, PBIC_DC2AC, QBIC, PESS_C, \
                PESS_DC, RESS, PMG
        else:
            from modelling.power_flow.idx_opf_set_points_tracing import PG, QG, RG, PUG, QUG, RUG, PBIC_AC2DC, PBIC_DC2AC, QBIC, PESS_C, \
                PESS_DC, RESS, PMG

        model["DG"]["COMMAND_PG"] = int(x[PG])
        model["DG"]["COMMAND_QG"] = int(x[QG])
        model["DG"]["COMMAND_RG"] = int(x[RG])

        model["UG"]["COMMAND_PG"] = int(x[PUG])
        model["UG"]["COMMAND_QG"] = int(x[QUG])
        model["UG"]["COMMAND_RG"] = int(x[RUG])

        model["BIC"]["COMMAND_AC2DC"] = int(x[PBIC_AC2DC])
        model["BIC"]["COMMAND_DC2AC"] = int(x[PBIC_DC2AC])

        model["BIC"]["COMMAND_Q"] = int(x[QBIC])

        model["ESS"]["COMMAND_PG"] = int(x[PESS_DC]) - int(x[PESS_C])
        model["ESS"]["COMMAND_RG"] = int(x[RESS])

        model["PMG"] = int(x[PMG])
        model["success"] = True # The obtained solution is feasible

    else:
        if model["COMMAND_TYPE"] is 0:
            from modelling.power_flow.idx_format_recovery import PG, QG, RG, PUG, QUG, RUG, PBIC_AC2DC, PBIC_DC2AC, QBIC, \
                PESS_C, PESS_DC, RESS, PMG, PPV, PWP, PL_AC, PL_UAC, PL_DC, PL_UDC
        else:
            from modelling.power_flow.idx_opf_set_points_tracing_recovery import PG, QG, RG, PUG, QUG, RUG, PBIC_AC2DC, PBIC_DC2AC, \
                QBIC,PESS_C, PESS_DC, RESS, PMG, PPV, PWP, PL_AC, PL_UAC, PL_DC, PL_UDC
        model["DG"]["COMMAND_PG"] = int(x[PG])
        model["DG"]["COMMAND_QG"] = int(x[QG])
        model["DG"]["COMMAND_RG"] = int(x[RG])

        model["UG"]["COMMAND_PG"] = int(x[PUG])
        model["UG"]["COMMAND_QG"] = int(x[QUG])
        model["UG"]["COMMAND_RG"] = int(x[RUG])

        model["BIC"]["COMMAND_AC2DC"] = int(x[PBIC_AC2DC])
        model["BIC"]["COMMAND_DC2AC"] = int(x[PBIC_DC2AC])
        model["BIC"]["COMMAND_Q"] = int(x[QBIC])

        model["ESS"]["COMMAND_PG"] = int(x[PESS_DC]) - int(x[PESS_C])
        model["ESS"]["COMMAND_RG"] = int(x[RESS])

        model["PMG"] = int(x[PMG])

        model["PV"]["COMMAND_CURT"] = int(model["PV"]["PG"]- x[PPV])
        # logger_uems.info("PV power curtailment amout {}".format(model["PV"]["PG"] - x[PPV]))
        model["WP"]["COMMAND_CURT"] = int(model["WP"]["PG"] - x[PWP])
        # logger_uems.info("Wind power curtailment amout {}".format(model["WP"]["PG"] - x[PWP]))
        model["Load_ac"]["COMMAND_SHED"] = int(model["Load_ac"]["PD"] - x[PL_AC])
        # logger_uems.info("Critical AC load shedding amout {}".format(model["Load_ac"]["PD"] - x[PL_AC]))
        model["Load_uac"]["COMMAND_SHED"] = int(model["Load_uac"]["PD"] - x[PL_UAC])
        # logger_uems.info("Non-critical AC load shedding amout {}".format(model["Load_uac"]["PD"] - x[PL_UAC]))
        model["Load_dc"]["COMMAND_SHED"] = int(model["Load_dc"]["PD"] - x[PL_DC])
        # logger_uems.info("Critical DC load shedding amout {}".format(model["Load_dc"]["PD"] - x[PL_DC]))
        model["Load_udc"]["COMMAND_SHED"] = int(model["Load_udc"]["PD"] - x[PL_UDC])
        # logger_uems.info("Non-critical DC load shedding amout {}".format(model["Load_udc"]["PD"] - x[PL_UDC]))

        model["success"] = False # The obtained solution is recovered

    return model
