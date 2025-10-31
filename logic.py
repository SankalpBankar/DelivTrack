# logic.py
def select_orders(order_ids, order_cost, max_capacity):
    """
    Selects orders under max capacity using a simple greedy algorithm.
    Orders are sorted by processing time (ascending) until capacity is filled.
    """
    sorted_orders = sorted(order_ids, key=lambda oid: order_cost[oid])
    selected = []
    total_cost = 0.0
    for oid in sorted_orders:
        cost = float(order_cost[oid])
        if total_cost + cost <= max_capacity:
            selected.append(oid)
            total_cost += cost
        else:
            break
    return selected, total_cost
