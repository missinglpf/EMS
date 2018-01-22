# The configuration file of local databases and network databases


# universal_database = \
#     {
#         "db_str" : 'mysql+pymysql://' + 'uems' + ':' + '2' + '@' + 'localhost' + '/' + 'universal_ems'
#     }
#ems_engine = create_engine('mysql://root:3333@localhost/localems')
local_database = \
    {
        "db_str" : 'mysql://root:3333@localhost/localems'
    }

history_database = \
    {
        "db_str" : 'mysql://root:3333@localhost/history_data'
    }
#
# local_history_database = \
#     {
#         "db_str" : 'mysql+pymysql://' + 'lems' + ':' + '3' + '@' + 'localhost' + '/' + 'history_data'
#     }
#
# weather_station_database = \
#     {
#         "db_str" : 'mysql+pymysql://' + 'lems' + ':' + '3' + '@' + 'localhost' + '/' + 'weather_station'
#     }

#IP address of local EMSs,
local_ems_ip_address = \
    {
        'module1':'192.168.1.150'
    }
