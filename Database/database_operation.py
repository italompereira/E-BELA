import time

import psycopg2
import psycopg2.extras
from Database.database_config import load_config


class DataBase:
    def __init__(self, spark_data):
       self.spark_data = spark_data

    def save(self):
        paths = [
            #['D:/Embeddings_concat/embeddings_literals'],
            #['D:/Embeddings_concat/embeddings_entities'],
            ['D:/Embeddings/embeddings_literals','D:/Embeddings/embeddings_entities_temp'],
            #['D:/Embeddings/embeddings_literals','D:/Embeddings/embeddings_entities'],
            ]
        tables = [
            #'df_embeddings_concat_intermediary',
            #'df_embeddings_concat',
            'df_embs_avg_interm_sent_trans',
            #'df_embeddings_average',
            ]

        for path, table in zip(paths, tables):
            df_embeddings = self.spark_data.load_Embeddings_path(path)
            #df_embeddings = df_embeddings.filter(df_embeddings.p.isNull())
            iterator = df_embeddings.toLocalIterator()
            count = 0
            inserted = 0
            tuples = []
            size = df_embeddings.count()
            for row in iterator:
                count += 1
                row_dict = row.asDict()
                s = row_dict['s'][0:255].replace("'", "''")
                p = None if row_dict['p'] is None else row_dict['p'][0:255].replace("'", "''")
                o = row_dict['o'][0:384].replace("'", "''")
                t = row_dict['typeObject']
                e = row_dict['embedding']

                tuples.append((s, p, o, t, e))

                if count == 30000:
                    start = time.time()
                    inserted += count
                    print(f'Inserted: {inserted}. Size {size}. Table: {table}')
                    self.insert(tuples,table)
                    count = 0
                    tuples = []
                    end = time.time()
                    print(end - start)
                    start = time.time()

                # if count == 1000:
                #     break

            inserted += len(tuples)
            print(f'Inserted: {inserted}. Size {size}. Table: {table}')
            self.insert(tuples,table)




    def insert(self, tuples, table):
        config = load_config()

        try:
            with  psycopg2.connect(**config) as conn:
                with  conn.cursor() as cur:
                    sql = f"""
                    INSERT INTO {table}(sub,pred,obj,type_obj,embedding) 
                    VALUES(%s,%s,%s,%s,%s) 
                    RETURNING id;"""

                    #work_mem = '4GB'
                    #cur.execute('SET work_mem TO %s', (work_mem,))
                    #maintenance_work_mem = '16GB'
                    #cur.execute('SET maintenance_work_mem TO %s', (maintenance_work_mem,))

                    # execute the INSERT statement
                    #cur.executemany(sql, tuples)
                    #psycopg2.extras.execute_batch(cur, sql, tuples)

                    argument_string = ",".join("('%s', '%s', '%s', '%s', '%s')" % (s, p, o, t, e) for (s, p, o, t, e) in tuples)
                    cur.execute("INSERT INTO {table}(sub,pred,obj,type_obj,embedding) VALUES".format(table=table) + argument_string)

                    # # get the generated id back
                    # rows = cur.fetchone()
                    # if rows:
                    #     id = rows[0]

                    # commit the changes to the database
                    conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            return id

    def get_top_k(self, vector, y_true, k):
        config = load_config()
        vector = str(vector) #str([item for item in vector])
        try:
            with psycopg2.connect(**config) as conn:
                table_name = 'df_embs_avg_interm_sent_trans'
                with conn.cursor() as cur:

                    # work_mem = '1GB'
                    # cur.execute('SET work_mem TO %s', (work_mem,))
                    # work_mem = '6GB'
                    # cur.execute('SET maintenance_work_mem TO %s', (work_mem,))

                    # sql = f"""
                    # SELECT sub, obj, embedding, embedding <-> %s as distance
                    # FROM {table_name}
                    # ORDER BY embedding <-> %s LIMIT %s;"""
                    # sql = f"""
                    # SELECT df.sub, df.obj, df.embedding, distance
                    # FROM {table_name} as df
                    # INNER JOIN (
                    #     SELECT sub, obj, embedding, embedding <-> %s as distance
                    #     FROM {table_name}
                    #     ORDER BY embedding <-> %s LIMIT %s
                    # ) as tmp ON tmp.sub = df.sub
                    # """

                    sql = f"""
                    SELECT df.sub, df.obj, df.embedding, df.embedding <=> %s as distance
                    FROM {table_name} as df
                    INNER JOIN (
                        SELECT distinct sub 
                        FROM (
                            SELECT sub, embedding
                            FROM {table_name} 
                            ORDER BY embedding <=> %s ASC 
                            LIMIT %s
                        ) as filter_table
                    ) as distinct_table ON distinct_table.sub = df.sub
                    """

                    cur.execute(sql, (vector, vector, k))
                    rows = cur.fetchall()

                    # Check if the grount truth is present on results
                    if not y_true in [row[0] for row in rows]:
                        with conn.cursor() as cur2:
                            sql = f"""
                            SELECT count(*) 
                            FROM {table_name} 
                            WHERE lower(sub) = lower(%s);"""

                            cur2.execute(sql, (y_true,))
                            result = cur2.fetchone()

                            if result[0] == 0:
                                return []
                            else:
                                with conn.cursor() as cur3:
                                    sql = f"""
                                    SELECT sub, obj, embedding, embedding <=> %s as distance 
                                    FROM {table_name}  
                                    WHERE (embedding <=> %s) < 1.0 
                                    ORDER BY distance LIMIT 10000;"""

                                    cur3.execute(sql, (vector, vector))
                                    rows_ = cur3.fetchall()#[:k]
                                    rows = list(set(sorted(rows_ + rows, key=lambda x: x[3]))) #[:k]


                    return rows #[(row[0], row[2]) for row in rows]


        except (Exception, psycopg2.DatabaseError) as error:
            print(error)