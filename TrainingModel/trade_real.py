def TradeRealWorld(symbol, device, leveragen,apikey,api_secret, saved=False, grad_lock=False):
    print("시간업데이트 중... 끄지 마셈")
    update_future_15min_csv(symbol)
    update_future_1hour_csv(symbol)
    update_future_1min_csv(symbol)
    print("시간업데이트 완료")

    # <---------------------------------------------------------------------->

    trader = Trader(device).to(device)
    client = Client(api_key=apikey, api_secret=api_secret)
    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_trader.pt'))

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])
    # <---------------------------------------------------------------------->

    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + '에 시작')
    hour = int(current_time[0:2])
    hourtemp = hour
    minute = int(current_time[3:5])
    minute_temp = minute

    # <---------------------------------------------------------------------->

    onehour = future_symbol_1hour_data(symbol).copy().iloc[-120:]
    onehour = build_numpy(onehour)
    s_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

    fifteen_data = future_symbol_15min_data(symbol).copy().iloc[-96:]
    fifteen_data = build_numpy(fifteen_data)
    s_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

    oneminute_data = future_symbol_1min_data(symbol).copy().iloc[-60:]
    oneminute_data = build_numpy(oneminute_data)
    s_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)

    sprime_oneH = s_oneH
    sprime_oneF = s_oneF
    sprime_oneM = s_oneM

    # <---------------------------------------------------------------------->

    selecter, fifteenminlocker = True, True

    # <---------------------------------------------------------------------->
    hidden = (
        torch.zeros([1, 1, 16], dtype=torch.float).to(device), torch.zeros([1, 1, 16], dtype=torch.float).to(device))
    h_in = [hidden, hidden, hidden]
    total_score = 0.0
    benefit = 100
    t = 0

    while True:
        sleep(0.2)
        # <---------------------------------------------------------------------->
        try:
            current_price = float(client.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])

        except:
            client = Client(api_key="", api_secret="")
            current_price = float(client.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])

        current_time = datetime.now().strftime("%H:%M:%S")

        hour = int(current_time[0:2])
        minute = int(current_time[3:5])

        # <---------------------------------------------------------------------->

        if hourtemp != hour:
            update_future_1hour_csv(symbol)
            hourtemp = hour
            onehour = future_symbol_1hour_data(symbol).copy().iloc[-120:]
            onehour = build_numpy(onehour)
            sprime_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

        if minute_temp != minute:
            fifteenminlocker = True
            update_future_1min_csv(symbol)
            minute_temp = minute
            oneminute_data = future_symbol_1min_data(symbol).copy().iloc[-60:]
            oneminute_data = build_numpy(oneminute_data)
            sprime_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)

        if minute % 15 == 0 and fifteenminlocker is True:
            fifteenminlocker = False
            update_future_15min_csv(symbol)
            fifteen_data = future_symbol_15min_data(symbol).copy().iloc[-96:]
            fifteen_data = build_numpy(fifteen_data)
            sprime_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

        # <---------------------------------------------------------------------->

        if selecter:
            with torch.no_grad():
                position_t, h_out = trader.SetPosition(s_oneH, s_oneF, s_oneM, h_in)

            # <---------------------------------------------------------------------->

            position_t = position_t.detach().reshape(-1)
            position_a = Categorical(position_t).sample().item()
            position_prob = position_t[position_a]

            # <---------------------------------------------------------------------->
            s_prime_select_1h = sprime_oneH
            s_prime_select_15m = sprime_oneF
            s_prime_select_1m = sprime_oneM

            s_select_1h = s_oneH
            s_select_15m = s_oneF
            s_select_1m = s_oneM

            selected_price = current_price * 0.9996

            # <---------------------------------------------------------------------->

            if position_a == 0:
                selecter = False
                position_v = 'LONG'
                best_reward = 0

            elif position_a == 1:
                selecter = False
                position_v = 'SHORT'
                best_reward = 0


            else:
                position_v = 'NONE'
                best_reward = 0
                selecter = False

            print(position_v + ': ' + str(current_price))

        # <---------------------------------------------------------------------->

        reward = (-100 + 100 * current_price / selected_price)

        if position_v == 'SHORT':
            reward *= -1

        # <---------------------------------------------------------------------->

        if position_v == 'NONE' and reward > 0:
            reward *= -1

        else:
            reward *= leveragen

        # <---------------------------------------------------------------------->

        reward_original = reward

        if reward > best_reward:
            best_reward = reward
        else:
            reward_original = reward
            reward -= best_reward

        # <---------------------------------------------------------------------->

        if reward_original < -1:
            reward_original = -1.0198  # -1.0198
            selecter = True

            print("{0}: total_benenfit is {1},and {2} reward is {3}".format(str(t), str(
                round(benefit, 2)), position_v, str(round(reward_original, 2))))

        # <---------------------------------------------------------------------->
        elif reward < -0.46:  # 큰거 -0.78 작은거: -0.46

            reward_original = (-100 + 99.96 * current_price / selected_price)

            if position_v == 'SHORT':
                reward_original *= -1

            if position_v == 'NONE' and reward > 0:
                reward_original *= -1

            else:
                reward_original *= leveragen

            if reward_original < -1:
                reward_original = -1.0198  # -1.0198
            else:
                reward_original /= 4

            print("{0}: total_benenfit is {1},and {2} reward is {3}".format(str(t), str(
                round(benefit, 2)), position_v, str(round(reward_original, 2))))
            selecter = True

        elif position_v is 'NONE' and reward_original < -0.1:  # 큰거 -0.1 작은거-0.05
            selecter = True

        # <---------------------------------------------------------------------->

        if selecter:
            if position_v is not 'NONE':
                benefit *= (1 + reward_original / 100)

            temp = reward_original

            if reward_original < 0:
                reward_original *= 5

            trader.TrainModelP(s_select_1h, s_select_15m, s_select_1m,
                               s_prime_select_1h, s_prime_select_15m, s_prime_select_1m, h_in, h_out,
                               position_a, position_prob, reward_original)

            reward_original = temp
            h_in = h_out
        t += 1
        if t % 150 == 0:
            print(reward_original)

        if total_score < benefit:
            total_score = benefit
            torch.save(trader.state_dict(), './model/' + symbol + '_trader.pt')

