from Repository.DatabaseConnection import DatabaseConnection


class CandlestickRepository:
    def __init__(self):
        self.__conn = DatabaseConnection().get_connection()
        self.__cursor = self.__conn.cursor()

    def get_all_candlestick(self):
        self.__cursor.execute(
            "SELECT c.Id, UPPER(c2.Symbol)  as MarketSymbol, c.PriceOpen, c.PriceHighest, c.PriceLowest , c.PriceClose , c.Volume, c.`DateTime`, WEEKDAY(c.`DateTime`) as DayWeek \
                                    FROM Candlestick c \
                                    inner join Market m on c.MarketId = m.Id \
                                    inner join Currency c2 on m.CurrencyBaseId  = c2.Id  \
                                    inner join Currency c3 on m.CurrencyQuoteId = c3.Id \
                       where c3.Symbol = 'brl'"
        )
        result = self.__cursor.fetchall()
        return result

    def get_candlestick_by_market_symbol(self, market_symbol):
        self.__cursor.execute(
            "SELECT c.Id, UPPER(c2.Symbol)  as MarketSymbol, c.PriceOpen, c.PriceHighest, c.PriceLowest , c.PriceClose , c.Volume, c.`DateTime`, WEEKDAY(c.`DateTime`) as DayWeek \
                                    FROM Candlestick c \
                                    inner join Market m on c.MarketId = m.Id \
                                    inner join Currency c2 on m.CurrencyBaseId  = c2.Id  \
                                    inner join Currency c3 on m.CurrencyQuoteId = c3.Id \
                       where c.`DateTime` > date(ADDDATE(NOW(), INTERVAL -3 MONTH)) \
                                    and c3.Symbol = 'brl' \
                                    and UPPER(c2.Symbol) = %s",
            (market_symbol.upper(),),
        )
        result = self.__cursor.fetchall()
        return result