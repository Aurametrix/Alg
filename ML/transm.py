def get_transition_matrix_emd(user, year):

    # lang2keep is the list of the 22 programming languages we kept
    # stairvalue() is the step function that comes from quantization

    # Build user's reactors for year and year+1
    reactors = zeros((len(lang2keep),2), order='f')

    for ind, code_in_lang in enumerate(dataset[user]):
        lang = code_in_lang[0]

        if lang in lang2keep:
            for y, qtt_coded_year in code_in_lang[1]:
                if y == year:
                    reactors[lang2keep.index(lang),0] = stairvalue(qtt_coded_year)

                elif y == year+1:
                    reactors[lang2keep.index(lang),1] = stairvalue(qtt_coded_year)

    if (sum(reactors[:,0]) == 0) or (sum(reactors[:,1]) == 0):
        # no transition to consider
        P = zeros((len(lang2keep), len(lang2keep)))
        return P

    else:
        # Normalization of reactors
        for i in [0, 1]:
            reactors[:,i] = [j/sum(reactors[:,i]) for j in reactors[:,i]]

    # compute the Earth Mover's distance between the 2 reactors thanks to the emd_with_flow function
    distance = np.ones((len(lang2keep), len(lang2keep)))
    dist, P = emd_with_flow(reactors[:,0], reactors[:,1], distance)
    P = np.asarray(P)

    return P
