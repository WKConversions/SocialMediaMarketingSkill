from fonts_sans_sheet import *
A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
WGT = {'inter': {'wght': 725, 'opsz': 32}, 'intertight': {'wght': 750}, 'manrope': {'wght': 800},
       'plusjakartasans': {'wght': 800}, 'instrumentsans': {'wght': 700, 'wdth': 100}, 'figtree': {'wght': 775},
       'geist': {'wght': 750}, 'onest': {'wght': 800}, 'archivo': {'wght': 775, 'wdth': 100},
       'schibstedgrotesk': {'wght': 750}, 'hankengrotesk': {'wght': 875}, 'albertsans': {'wght': 825},
       'publicsans': {'wght': 750}, 'hostgrotesk': {'wght': 800}, 'golostext': {'wght': 700},
       'arimo': {'wght': 700}, 'funnelsans': {'wght': 800}, 'familjengrotesk': {'wght': 700},
       'worksans': {'wght': 625}}
fams = list(WGT)
cands = [auto(f, WGT[f]) for f in fams]
json.dump([c[:1] + c[2:] for c in cands], open(A + 'cache_sans_round1.json', 'w'))
half = (len(cands) + 1) // 2
build(cands[:half], ['charging', 'and', 'yet'], A + 'fonts_sans_round1a.png', boxes={'charging': (740, 1176, 480, 615), 'and': (840, 1076, 480, 615), 'yet': (840, 1076, 480, 615)})
build(cands[half:], ['charging', 'and', 'yet'], A + 'fonts_sans_round1b.png', boxes={'charging': (740, 1176, 480, 615), 'and': (840, 1076, 480, 615), 'yet': (840, 1076, 480, 615)})
