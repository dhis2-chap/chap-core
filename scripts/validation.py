from chap_core.api import train_with_validation

#train_with_validation('FlaxModel', 'laos_full_data')
figs = train_with_validation('ProbabilisticFlaxModel', 'laos_full_data')
f = open('validation.html', 'w')
for fig in figs:
    f.write(fig.to_html())
f.close()
