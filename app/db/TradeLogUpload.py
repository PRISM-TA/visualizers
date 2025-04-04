def upload_trade_logs_to_database(session, trade_log_list):
    """
    Uploads a list of TradeLog objects into the database.
    
    :param trade_log_list: List[TradeLog] - A list of TradeLog objects to be uploaded.

    """
    try:
        # Use the session within a context manager to ensure it's properly closed after use
        with session() as session:
            # Add each TradeLog object to the session
            for trade_log in trade_log_list:
                session.add(trade_log)
            
            # Commit the changes to the database
            session.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise exception after logging

