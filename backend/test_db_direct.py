
import psycopg2
import sys

base_url = "postgresql://postgres:jPkGsaiCVrxjouATrCJPkcTPUiADpWzt@switchback.proxy.rlwy.net:32130/railway"
modes = ['', '?sslmode=require', '?sslmode=disable']

for mode in modes:
    url = base_url + mode
    print(f"\n--- Testing mode: {mode if mode else 'Default'} ---")
    try:
        conn = psycopg2.connect(url, connect_timeout=5)
        print(">> Connection SUCCESSFUL!")
        
        cur = conn.cursor()
        
        # Check users
        try:
            cur.execute('SELECT count(*) FROM "user";')
            count = cur.fetchone()[0]
            print(f">> User Count: {count}")
        except Exception as e:
            print(f">> Could not count users: {e}")
            
        conn.close()
        break # Stop if successful
        
    except Exception as e:
        print(f">> FAILED: {type(e).__name__} - {e}")
