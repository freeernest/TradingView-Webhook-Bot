from io import StringIO
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# FINALIZED_MODEL_SAV = '/home/ubuntu/TradingView-Webhook-Bot/finalized_model_tradingview.sav'
# SCALER_SAV = '/home/ubuntu/TradingView-Webhook-Bot/standardScaler.sav'
FINALIZED_MODEL_SAV = 'C:\\Users\\i506998\\Documents\\finalized_model_tradingview_checked_2000X14.sav'
SCALER_SAV = 'C:\\Users\\i506998\\Documents\\standardScaler.sav'

def predict(alertBody : str):

    filename = FINALIZED_MODEL_SAV
    clf_svm: SVC = pickle.load(open(filename, 'rb'))
    standardScaler: StandardScaler = pickle.load(open(SCALER_SAV, 'rb'))

    if(alertBody.__contains__('NaN')):
        return 0
    else:

        raw_data_list = alertBody.split(" ")
        # print(raw_data_list)
        raw_data_list = raw_data_list[1:]

        # formatting timestamp
        raw_data_list[raw_data_list.__len__()-1] = raw_data_list[raw_data_list.__len__()-1].split(')')[0]
        time = datetime.utcfromtimestamp(int(raw_data_list[raw_data_list.__len__()-1].replace(',', ''))/1000)
        formated_time = time.hour + time.minute / 60

        raw_data_list[raw_data_list.__len__()-1] = str(round(formated_time, 2))
        raw_data_list = list(map(lambda x: x.replace('NaN', '0'), raw_data_list))
        raw_data_list = list(map(lambda x: x.replace(',', ''), raw_data_list))
        # print(raw_data_list)
        alertBodyFormated = ','.join(raw_data_list)

        header = 'MACD(),MACDdiff,MACD().Avg,RSI(),"ExpAverage(close, length = 9)","ExpAverage(close, length = 21)",' \
                 '"ExpAverage(close, length = 34)","ExpAverage(close, length = 55)","ExpAverage(close, length = 88)",' \
                 '"ExpAverage(close, length = 100)",BollingerBands().UpperBand,BollingerBands().LowerBand,CCI(),' \
                 'StochasticFull().FullD,StochasticFull().FullK,imp_volatility,volume,close,GetTime()'

        alertBodyFormattedWithHeader = header + '\n' + alertBodyFormated

        array = pd.read_csv(StringIO(alertBodyFormattedWithHeader))
        # print('df:' + str(array))

        # df_large = pd.concat([df, df_no_missing])
        df_scaled = standardScaler.transform(array)
        # print('df_scaled:' + str(df_scaled))

        reshaped_scaled_row = df_scaled[0].reshape(1, -1)
        # print('reshaped_scaled_row: ' + str(reshaped_scaled_row))

        prediction = clf_svm.predict(reshaped_scaled_row)
        return prediction[0]


if __name__ == '__main__':
    # predict("MACDStratLE -0.2109576995 0.0400652161 -0.2510229157 57.6951291308 413.7833468816 412.9040915626 411.5029067591 409.0076627581 405.4664260435 404.3447981117 416.8000940679 409.0529059321 -34.1582009204 32.290846213 24.2380464714 22.29 36895 413.84 1659627900000")
    # print(str(predict("MACDStratLE -0.1372320549 0.0354370927 -0.1726691477 37.9122766829 454.3344457504 456.771361747 458.7664902308 461.121318504 463.544034569 NaN 461.9456690564 450.8803951156 -29.8233379786 28.3175346811 39.2151442038 22.78 500 454.37965929 1642593600000")))
    print(str(predict("MACDStratSE 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")))
    print(str(predict("MACDStratSE 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1")))
    print(str(predict("MACDStratLE 0.0742178891 0.0059301698 0.0682877193 70.4214575682 397.169585693 395.1500569687 393.1607478987 390.754452465 388.4348405568 387.870040573 399.4302082008 391.2707917992 92.9139825692 52.6298516801 52.4316230557 22.6 37475 398.71 1658501700000")))
    print(str(predict("MACDStratSE -0.0018868754 -0.0561396758 0.0542528004 64.2134487074 397.2636685544 395.3764154261 393.4167051617 391.000364877 388.6416980723 388.0635051161 399.4388199129 391.8801800871 53.0703720305 49.6941508228 30.2483052842 22.66 47872 397.64 1658502000000")))
    print(str(predict("MACDStratLE -0.392806696 0.0042462653 -0.3970529613 60.4552373472 393.8349838458 392.2221858418 390.4480834586 388.4257001926 386.5998472942 386.1757432951 395.9608415329 389.9471584671 14.6464646464 51.8219238615 67.770636438 24.04 26782 393.76 1658415000000")))
    print(str(predict("MACDStratLE -0.9313237327 0.0505870761 -0.9819108088 43.9038679286 384.9529865834 384.3301435707 383.5176660257 382.9484316185 382.5813751361 382.5293117388 389.8584121674 380.3745878325 -80.4468754718 43.94326634 9.9894023482 25.33 27669 382.31 1658174700000")))
    print(str(predict("MACDStratSE 0.0206369661 -0.0135381445 0.0341751106 67.8449784699 387.1208331707 384.8568110925 383.6645187648 382.9692024026 382.5677034855 382.5140256084 390.7889391173 378.3840608827 59.5461658842 57.7552222124 20.0443681122 24.59 15821 387.85 1658161800000")))
    print(str(predict("MACDStratSE 0.1726932336 -0.0414580808 0.2141513144 43.9838101077 380.2891880821 381.604476515 382.4539162399 382.7451249501 382.593548256 382.5825922914 387.2108455376 376.0691544624 -66.2648666025 60.2057465949 60.932448399 26.48 53133 380.25 1657740300000")))
    print(str(predict("MACDStratLE -0.0281983991 0.0680133307 -0.0962117297 43.1356014622 420.2017292236 422.204897099 423.888658329 425.2052065204 425.6366886953 425.5645289775 431.8000677341 413.4657924809 119.6059109368 41.4112561457 79.8850580347 21.21 139947 421.42511166 1626788700000")))
