from pydantic import BaseModel
from restack_ai.function import function, log
from src.functions.lookup_sales import lookupSales, LookupSalesInput

class PriceAlertInput(BaseModel):
    user_id: int
    item_id: int

@function.defn()
async def check_price_drop(input: PriceAlertInput) -> str:
    """
    Check if the item with item_id that the user is interested in
    has dropped in price beyond a certain discount threshold.
    """
    log.info(f"check_price_drop started with input: {input}")
    
    # Fetch all items on sale
    sales_data = await lookupSales(LookupSalesInput(category="any"))
    
    # Find the item in question
    item = next((i for i in sales_data.sales if i.item_id == input.item_id), None)
    if item:
        # Simple threshold check - for example, 10% discount
        if item.sale_discount_pct > 10:
            return f"Price drop alert: {item.name} is now ${item.sale_price_usd} (was ${item.retail_price_usd})."
        else:
            return f"No significant price drop for {item.name}."
    
    # If item wasn’t found or not on sale
    return "Item not found or not on sale."
