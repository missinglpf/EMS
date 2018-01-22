from apscheduler.schedulers.blocking import BlockingScheduler  # Time scheduler

import configuration.configuration_database as db_configuration  # The settings of databases
from sqlalchemy import create_engine  # Import database
from sqlalchemy.orm import sessionmaker
import schedule
from modelling import generators, loads, energy_storage_systems, convertors

from start_up import start_up_lems

from optimal_power_flow.main import short_term_operation

from utils import Logger
from keras.models import load_model

logger = Logger("Local_ems")
import time
def run():
    # Define the local models
    local_model_short = start_up_lems.start_up_lems.start_up()
    ACmodel = load_model('AC_PD_f.h5')
    PVmodel = load_model('PV_f.h5')
    # Set the database information
    db_str = db_configuration.local_database["db_str"]
    engine = create_engine(db_str, echo=False)
    Session = sessionmaker(bind=engine)
    session_lems_short = Session()

    # By short-term operation process
    logger.info("The short-term process in local ems starts!")

    #short_term_operation.short_term_operation_lems(local_model_short,session_lems_short)
    # sched_lems = BlockingScheduler()  # The schedulor for the optimal power flow
    # sched_lems.add_job( lambda: short_term_operation.short_term_operation_lems(local_model_short, session_lems_short, ACmodel),'cron', minute='*/1', second='*/5')  # The operation is triggered minutely
    # # sched_lems.add_job(short_term_operation.short_term_operation_lems, 'cron',
    # #                    args=(local_model_short,session_lems_short, ACmodel), minute='*/2',
    # #                    second='*/3')
    # # logger.info("The middle-term process in local EMS starts!")
    # # sched_lems.add_job(
    # #     lambda: middle_term_operation.middle_term_operation_lems(local_model_middle, socket_upload_ed, socket_download,
    # #                                                              info_ed,
    # #                                                              session_lems_middle),
    # #     'cron', minute='*/5', second='5')  # The operation is triggered every five minute
    # #
    # # logger.info("The long term process in local EMS starts!")
    # # sched_lems.add_job(
    # #     lambda: long_term_operation.long_term_operation_lems(local_model_long, socket_upload_uc, socket_download,
    # #                                                          info_uc,
    # #                                                          session_lems_long),
    # #     'cron', minute='*/30', second='30')  # The operation is triggered every half an hour
    # # sched_lems.start()
    #
    schedule.every().minute.do(lambda: short_term_operation.short_term_operation_lems(local_model_short,session_lems_short,ACmodel,PVmodel))
    while True:
        schedule.run_pending()
        time.sleep(1)
    # # t0 = time.time()
    #
    for i in range(1):
        # long_term_operation.long_term_operation_lems(local_model_long, socket_upload_uc, socket_download, info_uc,
        #                                                           session_lems_long)
        # middle_term_operation.middle_term_operation_lems(local_model_middle, socket_upload_ed, socket_download, info_ed,
        #                                                  session_lems_middle)
        short_term_operation.short_term_operation_lems(local_model_short,session_lems_short,ACmodel,PVmodel)
    #
    # print(time.time()-t0)


if __name__ == "__main__":
    ## Start the main process of local energy management system
    run()