import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eval_trend_acc(preds, labels):

    num_true = 0
    total = 0 #Only count the up or down trend
    for i in range(len(preds)):    
        if labels[i] == 1 or labels[i] == 2:
            total += 1
            if (preds[i][1] >= 0.5) or (preds[i][2] >=0.5):
                num_true += 1
    
    acc = num_true / total   
    # print("Accuracy on true trend {:.2f}".format(acc))     
    return acc, num_true, total

def eval_trend_return_acc(preds, gt, confidence_threshold = 0):    
    
    res = gt[preds > confidence_threshold]
    total = len(res)
    num_true = len(res[res > 0])
    if total > 0:
        acc = num_true / total
    else:
        acc = 0 

    mean = 0
    if len(res) != 0:
        mean = np.mean(res)

    return acc, num_true, total, mean

def buy_sell_trades(actual_price, predicted_trend, buying_threshold = 0.05, selling_threshold = 0.05, initial_budget = 100000000, trend_ahead = 5, trailing_stop = True):    
    
    hold_number_of_stocks = (int)(initial_budget / actual_price[0] / 100) * 100
    hold_money_left = initial_budget - hold_number_of_stocks * actual_price[0]  
    # The money got if buy and hold the stock
    budget_if_hold = initial_budget + hold_number_of_stocks * (actual_price[len(actual_price) - 1] - actual_price[0])

    #Calculate money if trade
    trade_number_of_stock = 0
    trade_budget = initial_budget
    bought_price = 0
    max_price = 0
    trade_budget_value = initial_budget

    budget_if_holds = []
    trade_budget_values = []
    buy_signal = []
    sell_signal = []
    
    for i in range(len(actual_price)):    
        
        # print("Day {}".format(i))
        # print(predicted_trend[i])

        action = 0  

        if trade_number_of_stock > 0:
            max_price = max(max_price, actual_price[i])
        
        if predicted_trend[i] > buying_threshold:
            
            action = 1            
            #If not have any, buy
            if trade_number_of_stock == 0:
                trade_number_of_stock = (int)(trade_budget / actual_price[i] / 100) * 100
                trade_budget -= actual_price[i] * trade_number_of_stock 
                bought_price = actual_price[i]
                max_price = actual_price[i]
                print("Buy {} stocks at price {}".format(trade_number_of_stock, bought_price))
                        
        elif predicted_trend[i] < selling_threshold:            
            action = 2                    
            if trade_number_of_stock > 0:                
                trade_budget += actual_price[i] * trade_number_of_stock
                profit = trade_number_of_stock * (actual_price[i] - bought_price)
                print("Sell {} stocks at price {} with profit {}".format(trade_number_of_stock, actual_price[i], profit))                
                trade_number_of_stock = 0
                bought_price = 0
                max_price = 0

        if actual_price[i] < max_price * (1 - 0.07):
            action = 2                    
            if trade_number_of_stock > 0:                            
                trade_budget += actual_price[i] * trade_number_of_stock
                profit = trade_number_of_stock * (actual_price[i] - bought_price)
                print("Sell {} stocks at price {} with profit {}".format(trade_number_of_stock, actual_price[i], profit))                
                trade_number_of_stock = 0
                bought_price = 0
                max_price = 0
                                            
        trade_budget_value = trade_budget + actual_price[i] * trade_number_of_stock
        trade_budget_values.append(trade_budget_value)
        budget_if_holds.append(hold_number_of_stocks * actual_price[i] + hold_money_left)
        
        if action == 1:
            buy_signal.append(trade_budget_value * 0.995) # For easier to read
            sell_signal.append(None)
            
        if action == 2:
            buy_signal.append(None)
            sell_signal.append(trade_budget_value * 1.005)
        
        if action == 0:
            buy_signal.append(None)
            sell_signal.append(None)
                
    #Money if we just bought as much at the start and sold near the end (Buy and hold)
    print("Budget if buy and hold:", budget_if_hold)  
    print("Budget if trade:", trade_budget_value)  

    gain_ratio = trade_budget_value / budget_if_hold
    sharp_ratio = trade_budget_value / initial_budget
    print("Gain ratio: {:.2f}".format(gain_ratio))  
    print("Sharp ratio: {:.2f}".format(sharp_ratio))  
    
    return trade_budget_values, budget_if_holds, buy_signal, sell_signal, gain_ratio, sharp_ratio

def trade_plot(data, symbol="Stock"):

    plt.figure(figsize=(16,6))
    plt.title(symbol)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Budget', fontsize=18)
    # plt.plot(train['close'])
    plt.plot(data[['hold_budget', 'trade_budget']])
    plt.plot(data["buy_signal"], '^', markersize=10, color='g')
    plt.plot(data["sell_signal"], 'v', markersize=10, color='r')
    plt.legend(['Buy and hold', 'Trade', "buy_signal", "sell_signal"], loc='upper left')
    plt.show()


def trade_analysis(actual_price, preds, symbol="Stock", buying_threshold = 0.5, selling_threshold = 0.5, trailing_stop = True, show_plot = True):

    assert len(actual_price) == len(preds)
    trade_budget_values, budget_if_holds, buy_signal, sell_signal, gain_ratio, sharp_ratio = buy_sell_trades(actual_price, preds, buying_threshold, selling_threshold, trailing_stop = trailing_stop)

    result = pd.DataFrame(data={"hold_budget" : budget_if_holds, "trade_budget" : trade_budget_values, "buy_signal" : buy_signal, "sell_signal" : sell_signal, "datetime" : actual_price.index})
    result.index = pd.DatetimeIndex(result['datetime'])

    if show_plot:
        trade_plot(result, symbol)
        
    return gain_ratio, sharp_ratio, result

    
if __name__ == '__main__':
    pass
    





    
    



