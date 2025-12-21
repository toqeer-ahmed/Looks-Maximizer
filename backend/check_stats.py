
import os
import sys
import argparse
from sqlalchemy import func

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_stats():
    # Lazy import to allow setting environment variable first
    try:
        from backend.app import app, db, User
    except ImportError as e:
        print(f"Error importing backend application: {e}")
        return

    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    
    # Mask password for display if present
    masked_uri = db_uri
    if "postgres" in db_uri and "@" in db_uri:
        try:
            part1 = db_uri.split("@")[1]
            prefix = db_uri.split("@")[0].split(":")[0]
            masked_uri = f"{prefix}:****@{part1}"
        except:
            pass
            
    print("\n" + "="*50)
    print(f"  LOOKS MAXIMIZER - DATABASE STATISTICS")
    print("="*50)
    print(f"Database Source: {masked_uri}")
    
    if 'sqlite' in db_uri:
        print("[INFO] Showing LOCAL development data.")
        print("To see PRODUCTION data, provide your Railway Database URL.")
    else:
        print("[INFO] Connected to PRODUCTION database.")
    
    print("-" * 50)

    with app.app_context():
        try:
            # Total Signups
            total_users = User.query.count()
            print(f"\nTotal Registered Users: {total_users}")
            
            # Premium Users
            pro_users = User.query.filter(User.plan != 'free').all()
            print(f"Total Premium Users:    {len(pro_users)}")
            
            print("-" * 50)
            if pro_users:
                print("Premium Plan Breakdown:")
                plan_counts = db.session.query(User.plan, func.count(User.plan)).group_by(User.plan).all()
                for plan, count in plan_counts:
                    print(f"  • {plan.capitalize()}: {count}")
            else:
                print("No premium subscriptions active (All users are on 'Free' plan).")
            print("="*50 + "\n")
                
        except Exception as e:
            print(f"\nError connecting to database: {e}")
            print("Please check if your IP is allowed or if the connection URL is correct.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check database statistics.')
    parser.add_argument('--url', type=str, help='Railway Database URL (postgres://...)')
    args = parser.parse_args()

    if args.url:
        os.environ['DATABASE_URL'] = args.url
    
    check_stats()
