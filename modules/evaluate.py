aspect_name = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
# aspect_name = ['cấu hình','mẫu mã','hiệu năng','ship','giá','chính hãng','dịch vụ','phụ kiện']

def cal_aspect_prf(goldens, predicts, num_of_aspect, verbal=False):
    """

    :param verbal:
    :param num_of_aspect:
    :param list of models.AspectOutput goldens:
    :param list of models.AspectOutput predicts:
    :return:
    """
    tp = [0] * num_of_aspect
    fp = [0] * num_of_aspect
    fn = [0] * num_of_aspect

    for g, p in zip(goldens, predicts):
        for i in range(num_of_aspect):
            if g[i] == p[i] == 1:
                tp[i] += 1
            elif g[i] == 1:
                fn[i] += 1
            elif p[i] == 1:
                fp[i] += 1
    print('tp fn fp')
    for i in range(num_of_aspect):
        print(tp[i], fn[i], fp[i], aspect_name[i])

    p = [tp[i]/(tp[i]+fp[i]) for i in range(num_of_aspect)]
    r = [tp[i]/(tp[i]+fn[i]) for i in range(num_of_aspect)]
    f1 = [2*p[i]*r[i]/(p[i]+r[i]) for i in range(num_of_aspect)]

    micro_p = sum(tp)/(sum(tp)+sum(fp))
    micro_r = sum(tp)/(sum(tp)+sum(fn))
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)

    macro_p = sum(p)/num_of_aspect
    macro_r = sum(r)/num_of_aspect
    macro_f1 = sum(f1)/num_of_aspect

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)
        print('micro:', (micro_p, micro_r, micro_f1))
        print('macro:', (macro_p, macro_r, macro_f1))

    # print('p:')
    # for element in p:
    #     print(element)
    # print("")
    #
    # print('r:')
    # for element in r:
    #     print(element)
    # print("")
    #
    # print('f1:')
    # for element in f1:
    #     print(element)
    # print("")
    #
    # print('micro:')
    # print(micro_p)
    # print(micro_r)
    # print(micro_f1)
    # print("")
    #
    # print('macro:')
    # print(macro_p)
    # print(macro_r)
    # print(macro_f1)
    # print("")

    return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)
