from Repository.DatabaseConnection import DatabaseConnection

class MarketRepository:
    def __init__(self):
        self.__conn = DatabaseConnection().get_connection()
        self.__cursor = self.__conn.cursor()

    def get_all_markets(self):
        self.__cursor.execute(
            " SELECT m.*, bas.Symbol as CurrencyBaseSymbol, quo.Symbol as CurrencyQuoteSymbol \
                    FROM Market m \
                    inner join Currency bas on m.CurrencyBaseId  = bas.Id\
                    inner join Currency quo on m.CurrencyQuoteId = quo.Id \
                    WHERE quo.Symbol = 'brl'"
        )
        result = self.__cursor.fetchall()
        return result