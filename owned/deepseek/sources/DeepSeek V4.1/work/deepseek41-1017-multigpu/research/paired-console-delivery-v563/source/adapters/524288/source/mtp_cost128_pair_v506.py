

def initial_costs(context):
    # One measured CPUv118/own3/cache4 case, an experimental cost prior.
    # This changes prefix length only; every emitted token is target-verified.
    if context!=131072:raise ValueError('v128 cost prior admits only128k own3/cache4 experiment')
    return dict(verify_seconds_by_inputs={'1': 0.03408946499985177, '3': 0.04993215099966619, '5': 0.08767888500005938, '2': 0.043931956504820846, '4': 0.08455306199903134, '6': 0.08294724050210789},commit_seconds=.00065)

