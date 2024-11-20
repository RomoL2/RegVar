import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import RBPamp.report
from RBPamp import dominguez_rbps as dom_rbps
# dom_rbps = "BOLL,CELF1,CNOT4,CPEB1,DAZ3,DAZAP1,EIF4G2,ELAVL4,ESRP1,EWSR1,FUBP1,FUBP3,FUS,A1CF,HNRNPA1,HNRNPA2B1,HNRNPC,HNRNPCL1,HNRNPD,HNRNPDL,HNRNPF,HNRNPH2,HNRNPK,HNRNPL,IGF2BP1,IGF2BP2,ILF2,KHDRBS2,KHDRBS3,KHSRP,MBNL1,MSI1,NOVA1,NUPL2,PABPN1L,PCBP1,PCBP2,PCBP4,PRR3,PTBP3,PUF60,PUM1,RALY,RBFOX2,RBFOX3,RBM15B,RBM22,RBM23,RBM25,RBM4,RBM41,RBM45,RBM4B,RBM6,RBMS2,RBMS3,RC3H1,SF1,SFPQ,SNRPA,SRSF10,SRSF11,SRSF2,SRSF4,SRSF5,SRSF8,SRSF9,TARDBP,TIA1,TRA2A,TRNAU1AP,UNK,ZCRB1,ZFP36,ZNF326".split(',')

def _domain(dom):
    if type(dom) == float:
        return 'NA'
    from collections import defaultdict
    ds = defaultdict(int)
    for d in dom.split(','):
        ds[d] += 1
    tbl = np.array(sorted([(v,k) for k,v in list(ds.items())]))[::-1]
    if len(list(ds.keys())) == 1:
        return list(ds.keys())[0]
        # return "{} {}".format(ds.values()[0], ds.keys()[0]) # only one domain type
    else:
        return "mixed"
    # return "+".join(sorted(ds.keys()))
    # return tbl[0][1]

def _domain_count(dom):
    if type(dom) == float:
        return 'NA'
    n = len(dom.split(','))
    if n < 4:
        return str(n)
    else:
        return "4+"

def _topR(v):
    if v > 20.:
        return '20+'
    elif v > 5:
        return '5-20'
    elif v > 2:
        return '2-5'
    elif v > 1:
        return '1-2'
    else:
        return 'NA'

# linscore = pd.read_table('scores.txt', header=None, names=['rbp','linscore'])
# fp_params = pd.read_table('opt_a.txt', header=None, names=['rbp', 'acc_scale'])
extra = pd.read_table('domains.txt', header=None, names=['rbp', 'domains'])
extra['domain'] = extra['domains'].apply(_domain)
extra['n_dom'] = extra['domains'].apply(_domain_count)

topR = pd.read_table('topR.txt', header=None, names=['rbp', 'top_R'])
# print topR.describe()
extra = extra.merge(topR, on='rbp')
# print extra.describe()
extra['max_R'] = extra['top_R'].apply(_topR)

domain_order=['RRM', 'KH', 'ZNF', 'mixed', 'other']
# df = pd.read_table('/home/mjens/engaging/CI_results.tsv')
# df = pd.read_table('std.72.tsv')
# df = pd.read_table('std.tsv')
df = pd.read_table(sys.argv[1])

df = df.merge(extra, on='rbp')
out = df.query('rbp in @dom_rbps')
out['fold_error'] = out['nostruct__err_initial'] / out['full__err_final']
out.to_csv('fit_results.csv', sep='\t')
sys.exit(0)

pf = df[ ['rbp', 'nostruct__err_initial', 'full__err_final', 'full__best_corr', 'domain', 'top_R', 'max_R']].query('rbp in @dom_rbps')
pf['fold_error'] = pf['nostruct__err_initial'] / pf['full__err_final']
x = pf[['rbp', 'fold_error']]
print((x.sort_values('fold_error')))
print(("Final MEAN CORRELATION", pf['full__best_corr'].mean()))
print(("Initial mean LOG ERROR", np.log10(df.query('rbp in @dom_rbps')['full__err_initial'].values).mean()))
print(("Final mean LOG ERROR", np.log10(df.query('rbp in @dom_rbps')['full__err_final'].values).mean()))
print((pf.query('rbp == "HNRNPL"')))
# print pf


plt.figure(figsize=(3, 1.5))
bpcorr = sns.boxplot(
    data=pf,
    y='domain',
    x='full__best_corr',
    order=domain_order,
    width=.5,
    palette='viridis',
    orient='h',
    flierprops = dict(marker='o')
)
# bpcorr.set_xticklabels(bpcorr.get_xticklabels(), rotation=90)

print((pf.groupby(['domain'])[ ['domain', 'full__best_corr'] ]))
pf_exc = pf.query("rbp not in ['ZNF326', 'RC3H1', 'RBM22']")
medians = pf_exc.groupby(['domain'])['full__best_corr'].median().values
print(("median correlations excluding ZNF326, RBM22, and RC3H1", medians, pf_exc['full__best_corr'].median()))
folds = pf.groupby(['domain'])['fold_error'].median().values, pf['fold_error'].median()
print(("median fold error reductions", folds))
# for dom, med in zip(domain_order, )
print(pf.query('rbp == "TRA2A"'))
plt.tight_layout()
sns.despine()
plt.savefig('fit_corr.pdf')
plt.close()

plt.figure(figsize=(1.5,3))
bperr = sns.boxplot(
    data=pf,
    x='domain',
    y='fold_error',
    order=domain_order,
    width=.5,
    palette='viridis',
    orient='v',
    flierprops = dict(marker='o'),
    linewidth = .5,
    fliersize = 1,
)
bperr.set_xticklabels(bperr.get_xticklabels(), rotation=90)

plt.tight_layout()
sns.despine()
plt.savefig('fit_err.pdf')
plt.close()

pf['log_R'] = np.log2(pf['top_R'])
lmp = sns.lmplot(
    data=pf,
    y='full__best_corr',
    x='log_R',
    fit_reg=False,
    hue='domain',
    hue_order=domain_order,
    legend=False,
    palette='viridis',
    markers=['o', '^', 's', '*', '>']
)
plt.legend(loc='lower right')
plt.gcf().set_size_inches(3, 3)

# rbp_annotate = ['HNRNPL', 'ZNF326', 'PUM1', 'RBFOX2', 'MBNL1', 'SRSF2', 'TIA1']
# ann_ofs = {
#     'HNRNPL' : (-.1, 0),
#     'ZNF326' : (.0, 1),
#     'MBNL1' : (0, 1),
#     'SRSF2' : (.05, 1),
# }
# for x in pf.query('rbp in @rbp_annotate').itertuples():
#     print x
#     dx, dy = ann_ofs.get(x.rbp, (-.1, 0) )
#     plt.annotate(
#         x.rbp,
#         xy=(x.full__best_corr, x.fold_error),
#         arrowprops=dict(arrowstyle='->'),
#         xytext=(x.full__best_corr + dx, x.fold_error + dy)
#     )

# plt.xlabel("fold model error reduction")
plt.xlabel("max 6-mer R-value (log2)")
plt.ylabel("max 6-mer correlation after fit")
plt.ylim(0.5, 1)
print((pf[['rbp', 'full__best_corr', 'top_R', 'max_R']].sort_values('top_R', ascending=False)))
# plt.ylim(1, pf['fold_error'].max()+.5)
# sns.despine()
plt.tight_layout()
plt.savefig('fit_qual.pdf')
plt.close()



fig, ax = plt.subplots(figsize=(4,2))
ax.set_xlim(0.5, 1)
dom_order = ['KH', 'ZNF', 'RRM', 'mixed', 'other']
ax = sns.swarmplot(
    ax=ax,
    orient='h',
    data=pf,
    size=4.,
    x='full__best_corr', 
    y='domain', 
    order = dom_order,
    hue='max_R', 
    hue_order = ['20+','5-20','2-5','1-2'],
    palette='viridis_r',
)

for dom in dom_order:
    select = pf[pf['domain'] == dom]
    print(select.sort_values('full__best_corr')[['domain', 'rbp', 'full__best_corr']])

grouped = pf.groupby('domain')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print((pf[pf.full__best_corr == pf[pf.domain == 'other'].full__best_corr.min()]))
means_df = grouped.median()
print(">>>>>MEDIANS")
print(means_df)
# print((pf[pf.rbp == 'RBM22']))
means = means_df['full__best_corr'].loc[dom_order]
ax.plot(means, np.arange(len(dom_order)), '|r', zorder=np.inf, markersize=15, markeredgewidth=1.5, solid_capstyle='round', dash_capstyle='round')
sns.despine()
plt.tight_layout()
plt.savefig('overview_swarm.pdf')





fig, ax = plt.subplots(figsize=(4,2))
# ax.set_xlim(0.5, 1)
dom_order = ['KH', 'ZNF', 'RRM', 'mixed', 'other']
ax.set_xscale('log')
ax = sns.swarmplot(
    ax=ax,
    orient='h',
    data=pf,
    size=4.,
    x='fold_error', 
    y='domain', 
    order = dom_order,
    hue='max_R', 
    hue_order = ['20+','5-20','2-5','1-2'],
    palette='viridis_r',
)

for dom in dom_order:
    select = pf[pf['domain'] == dom]
    print(select.sort_values('fold_error')[['domain', 'rbp', 'fold_error']])

grouped = pf.groupby('domain')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print((pf[pf.fold_error == pf[pf.domain == 'other'].fold_error.min()]))
means_df = grouped.median()
print(">>>>>MEDIANS")
print(means_df)
# print((pf[pf.rbp == 'RBM22']))
means = means_df['fold_error'].loc[dom_order]
ax.plot(means, np.arange(len(dom_order)), '|r', zorder=np.inf, markersize=15, markeredgewidth=1.5, solid_capstyle='round', dash_capstyle='round')
sns.despine()
plt.tight_layout()
plt.savefig('overview_swarm_error.pdf')










sys.exit(0)

print((df[ ['rbp', 'full__err_perc', 'full__err_perc', 'full__corr_inc', 'full__corr_inc'] ].describe()))

print(("KHDR2", df.query('rbp == "KHDR2"')[['full__best_corr', 'full__best_corr']]))
print(("failed nostruct optimization", df[ df['full__err_perc'] > 99.]['rbp']))
print(("failed full optimization", df[ df['full__err_perc'] > 99.]['rbp']))


plt.figure()
plt.scatter(df['full__best_corr'], df['full__best_corr'])
print(("worse with structure", df.query('full__best_corr > full__best_corr + .1')['rbp']))
plt.xlim(0.5, 1)
plt.ylim(0.5, 1)
plt.xlabel("max 6-mer correlation after seq. only optimization")
plt.ylabel("max 6-mer correlation after footprint optimization")
plt.plot([.5,1.],[.5,1.], 'k', linestyle='dashed', linewidth=.5)
# plt.scatter(df['full.err_perc'], .01*df['full.err_perc']*df['full.err_perc'])
# plt.xlim(0, 105)
# plt.ylim(0, 105)
plt.xlabel("% error after seq. only optimization")
plt.ylabel("% error after footprint optimization")
# n = len(df)
# plot_df = pd.DataFrame.from_dict(dict(
#     opt = (['seqonly',] * n) + (['full',] * n),
#     err_perc = pd.concat(
#         [
#             df['full.err_perc'],
#             df['full.err_perc'],
#         ]
#     )
# ))
# ax = sns.boxplot(x='opt', y='err_perc', data=plot_df, order=['seqonly', 'full'], width=.5)  #, hue_order=['full', 'full'])
# # ax = sns.swarmplot(x='domain', y=col, data=df, order=order, linewidth=1, hue='run', dodge=True)  #, dodge=True, hue_order=['full', 'full'])
# #     # lmp = sns.lmplot(x='rerr'    df['domain'] = df['domains'].apply(_domain)rr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
# #     plt.xlabel('RBD type')
# #     plt.ylabel('final 6mer correlation')
# #     plt.ylim(.25,1.)

sns.despine(trim=True)
plt.savefig('err_perc.pdf')
plt.close()

sys.exit(0)




# print df.describe()

# print fp_params.describe()
# print fp_params
# df = linscore.join(descent, on='rbp', rsuffix='_lin')

def _linearity(v):
    if v > 1.:
        return '1+'
    elif v > .7:
        return '0.7-1'
    elif v > 0:
        return '0.7-'
    else:
        return "NA"

def _scale(v):
    if v > .5:
        return '0.5+'
    elif v > .2:
        return '0.2-0.5'
    elif v > .1:
        return '0.1-0.2'
    elif v > 0.05:
        return '0.05-0.1'
    else:
        return '< 0.05'



def load_descent_run(fname, run='full'):
    # descent = pd.read_table('descent2.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
    # descent = pd.read_table('descent_bugfix_noopt.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
    df = domains.merge(pd.read_table('/home/mjens/engaging/ci_ns.tsv'), on='rbp')
    df = df.merge(topR, on='rbp')
    # descent = pd.read_table(fname, header=None, names=['rbp','rerr','ferr','corr','steps'])

    # df = descent.merge(linscore, how='left', on='rbp')
    # df = df.merge(domains, how='left', on='rbp')
    # df = df.merge(topR, how='left', on='rbp')
    # df = df.merge(fp_params, how='left', on='rbp')
    # df['motif_linearity'] = df['linscore'].apply(_linearity)
    # df['domain'] = df['domains'].apply(_domain)
    # df['n_dom'] = df['domains'].apply(_domain_count)
    # df['opt_a'] = df['acc_scale'].apply(_scale)
    df['run'] = run

    return df.set_index('rbp')
    # return df

def label_point(row):
    x,y,val = row
    plt.text(x, y-0.02, str(val))

# print intersect.describe()
# print intersect[['corr_full', 'corr_nostruct']]
# sys.exit(0)
# print df
def by_R_value_plot(df):
    order = ['20+','5-20','2-5','1-2']
    ax = sns.boxplot(x='max_R', y="nostruct.best_corr", data=df, order=order[::-1], whis=np.inf, width=.5, hue='run', dodge=True)  #, hue_order=['full', 'full'])
    ax = sns.swarmplot(x='max_R', y="nostruct.best_corr", data=df, order=order[::-1], color=".2", hue='run', dodge=True)  #, hue_order=['full', 'full'])
    plt.xlabel('max. observed 6-mer R-value')
    plt.ylabel('max. R-value correlation after fit')
    plt.ylim(.25, 1)
    plt.savefig('overview.pdf')
    plt.close()

def by_domain_plot(df, col="nostruct.best_corr"):
    # order = ['RRM','RRM+KH','    df = descent.merge(linscore, how='left', on='rbp')F', 'RRM+other', 'KH', 'KH+ZNF','KH+other', 'ZNF', 'ZNF+other','other','NA']
    # order = ['1 RRM', '2 RRM'    df = df.merge(domains, how='left', on='rbp')RM', '4 RRM', '1 KH', '2 KH', '3 KH', '4 KH', '1 ZNF', '2 ZNF', '3 ZNF', '4 ZNF', '1 other', 'mixed']
    order = ['RRM', 'KH', 'ZNF', 'other', 'mixed']
    ax = sns.boxplot(x='domain', y=col, data=df, order=order, whis=np.inf, width=.5, hue='run')  #, hue_order=['full', 'full'])
    ax = sns.swarmplot(x='domain', y=col, data=df, order=order, linewidth=1, hue='run', dodge=True)  #, dodge=True, hue_order=['full', 'full'])
    # lmp = sns.lmplot(x='rerr'    df['domain'] = df['domains'].apply(_domain)rr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    plt.xlabel('RBD type')
    plt.ylabel('final 6mer correlation')
    plt.ylim(.25,1.)
    sns.despine(trim=True)
    plt.savefig('fit_by_domain_type.pdf')
    plt.close()

    order = ['1', '2', '3', '4+']
    ax = sns.boxplot(x='n_dom', y=col, data=df, order=order, whis=np.inf, width=.5, hue='run')  #, hue_order=['full', 'full'])
    ax = sns.swarmplot(x='n_dom', y=col, data=df, order=order, linewidth=1, hue='run', dodge=True)  #, hue_order=['full', 'full'])
    # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    plt.xlabel('RBD number')
    plt.ylabel('final 6mer correlation')
    plt.ylim(.25,1.)
    sns.despine(trim=True)

    plt.savefig('fit_by_domain_num.pdf')
    plt.close()

def scale_by_domain(df):
    order = ['RRM', 'KH', 'ZNF', 'other', 'mixed']
    ax = sns.boxplot(x='domain', y="acc_scale", data=df, order=order, whis=np.inf, width=.5)
    ax = sns.swarmplot(x='domain', y="acc_scale", data=df, order=order, color=".2", dodge=True)

    # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    plt.xlabel('RBD type')
    plt.ylabel('accessibility scale')
    plt.savefig('scale_by_domain_type.pdf')
    plt.close()

    # order = ['1', '2', '3', '4+']
    # ax = sns.boxplot(x='n_dom', y="corr", data=df, order=order, whis=np.inf, width=.5, hue='run', hue_order=['full', 'full'])
    # ax = sns.swarmplot(x='n_dom', y="corr", data=df, order=order, color=".2", hue='run', dodge=True, hue_order=['full', 'full'])

    # # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    # plt.xlabel('RBD number')
    # plt.ylabel('final 6mer correlation')
    # plt.savefig('fit_by_domain_num.pdf')
    # plt.close()

# df = load_descent_run(sys.argv[1])
# df_nostruct = load_descent_run(sys.argv[2], run='full')
# combined = df.append(df_nostruct)

# intersect = df.join(df_nostruct, lsuffix="_full", rsuffix='_nostruct', how='inner')
df = extra.merge(pd.read_table('/home/mjens/engaging/ci_ns.tsv'), on='rbp')
df['run'] = 'mutli PSAM (CI)'

df2 = extra.merge(pd.read_table('/home/mjens/engaging/1m_ns.tsv'), on='rbp')
df2['run'] = 'single PSAM'

by_domain_plot(pd.concat([df, df2]))
by_R_value_plot(pd.concat([df, df2]))
sys.exit(0)

scale_by_domain(df)

def corr_scatter(full, nostruct):
    df = intersect
    # print df[['rbp_full', 'rbp_nostruct', 'corr_full', 'corr_nostruct']]
    order = ['20+','5-20','2-5','1-2'][::-1]
    order = ['0.5+', '0.2-0.5', '0.1-0.2', '0.05-0.1', '<0.05'][::-1]
    # lmp = sns.lmplot(x='corr_nostruct',y='corr_full',data=df, fit_reg=False, hue='max_R_full', hue_order=order[::-1], legend=True,palette='viridis')
    # order = ['1+', '0.7-1', '0.7-', "NA"]
    # lmp = sns.lmplot(x='corr_nostruct',y='corr_full',data=df, fit_reg=False, hue='max_R_full', hue_order=order[::-1], legend=False, palette='viridis')
    print((df.describe()))
    lmp = sns.lmplot(x='corr_nostruct',y='corr_full',data=df, fit_reg=False, hue='opt_a_full', hue_order=order[::-1], legend=False, palette='viridis')

    df['delta'] = df['corr_full'] - df['corr_nostruct']
    # df.set_index(['rbp','delta', 'corr_full','corr_nostruct'])
    # what are the proteins with highest difference?
    by_delta = df.sort_values('delta', ascending=False)

    def label_point(row, ax):
        print(row)
        print((row.Index))

        ax.text(row.corr_nostruct+.02, row.corr_full, str(row.Index))

    # print by_delta[['rbp_full', 'delta']][:5]
    # print by_delta
    for row in by_delta[['corr_full','corr_nostruct']][:3].itertuples():
        label_point(row, plt.gca())

    # for row in by_delta[['corr_full','corr_nostruct']][-5:].itertuples():
    #     label_point(row, plt.gca())

    by_corr = df.sort_values('corr_nostruct', ascending=False)
    # print by_corr.iloc[0]
    for row in by_corr[['corr_full', 'corr_nostruct']][:1].itertuples():
        label_point(row, plt.gca())

    for row in by_corr[['corr_full', 'corr_nostruct']][-5:].itertuples():
        label_point(row, plt.gca())

    # print by_delta[['rbp_full', 'delta']][-5:]
    # for row in by_delta.iloc[-5:, :].iterrows():
    #     label_point(row, plt.gca())


    plt.xlabel('max. Pearson-R nostruct model')
    plt.xlim(0.3,1.1)
    plt.ylim(0.3,1.1)
    plt.plot([0,1],[0,1],'-k', linewidth=.1)
    plt.ylabel('max. Pearson-R full model')
    # tolabel.apply(label_point, axis=1)
    # plt.legend(loc='upper left', title="max. 6-mer enrichment")
    plt.legend(loc='upper left', title="accessibility scale")
    plt.savefig('overview_scatter.pdf')
    plt.close()


corr_scatter(df, df_nostruct)
by_R_value_plot(combined)
by_domain_plot(combined)

# print "worst proteins using nostruct"
# df = df_nostruct.sort_values('corr')
# print df[:10]
# rbps = ['RBFOX3','RBFOX2','ELAVL4', 'HNRNPA0','GST','SRSF11','SRSF5','SRSF4','SRSF2','GST','PRR3','ESRP1', 'MSI1']
# rows = df.loc[df['rbp'].isin(rbps)]
# tolabel = rows[['top_R','corr','rbp']]

def quality(df):
    corr = df['corr']
    qrange = np.percentile(corr,[25,75])
    corrmean = np.mean(corr)
    corrmedian = np.median(corr)
    print(("final best R-value correlation quartile range, mean, median", qrange, corrmean, corrmedian))

print("no structure")
quality(df_nostruct)
print("full model")
quality(df)
# lmp = sns.lmplot(x='top_R',y='corr',data=df, fit_reg=False, hue='motif_linearity', hue_order=['1+','0.7-1','0.7-','NA'],legend=True,palette='viridis')
# # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', legend=True)
# lmp.set(xscale="log")
# tolabel.apply(label_point, axis=1)

# plt.xlabel('highest 6mer R-value')
# plt.ylabel('final 6mer correlation')
# plt.savefig('overview_linearity.pdf')

# plt.close()


# lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='max_R', hue_order=order,legend=True, palette='husl')
# tolabel.apply(label_point, axis=1)

# plt.xlabel('final relative 6mer error')
# plt.ylabel('final 6mer correlation')
# plt.savefig('overview_max_R.pdf')
# plt.close()



