import time

import psycopg2
import psycopg2.extras
from Database.database_config import load_config
from pyspark.sql.functions import *
from pyspark.sql.window import Window

class DataBase:
    def __init__(self, spark_data, model:'ST', weighted:False, top_n=None, top_n_way=None, config_experiment=''):
        self.spark_data = spark_data
        self.model = model
        self.weighted = weighted
        self.top_n = top_n
        self.top_n_way = top_n_way
        self.config_experiment = config_experiment

    def save(self):
        print(f'Saving at database: {self.config_experiment}. ')

        paths = (
            [
                self.spark_data.PATH_EMBEDDINGS_LITERALS + f'{self.model}_{("WGT" if self.weighted else "NotWGT")}',
                self.spark_data.PATH_EMBEDDINGS_ENTITIES_TEMP + self.config_experiment
            ],
            # [
            #     self.spark_data.PATH_EMBEDDINGS_LITERALS + f'{self.model}_{("WGT" if self.weighted else "NotWGT")}',
            #     self.spark_data.PATH_EMBEDDINGS_ENTITIES + self.config_experiment
            # ],

        )

        tables = [
            f'df_embs_{str.lower(self.config_experiment)}',
            #f'df_embs_{str.lower(self.config_experiment)}'
        ]

        for path, table in zip(paths,tables):

            df_emb_literals = self.spark_data.load_Embeddings_path(path[0])
            if self.top_n is not None:
                # Group literals and count
                df_emb_literals = df_emb_literals.join(
                    df_emb_literals.groupBy(lower("o")).count().withColumnRenamed("lower(o)", "o_"),
                    (lower(col('o')) == lower(col('o_')))
                ).drop('o_')

                if self.top_n_way == 'SIMPLE':
                    # Enumerate each line per group of entity
                    window_sub = Window.partitionBy(*["s"]).orderBy(["s", "count"])
                    df_emb_literals = df_emb_literals.withColumn("top_n", row_number().over(window_sub))

                elif self.top_n_way == 'GROUPED':
                    # Gets every distinct entity and count
                    df_emb_literals_aux = df_emb_literals.select('s', 'count').withColumnRenamed("s", "s_").withColumnRenamed("count", "count_").distinct()

                    # Enumerate each line per group of entity and literals with the same quantity
                    window_sub = Window.partitionBy(*["s_"]).orderBy(["s_", "count_"])
                    df_emb_literals_aux = df_emb_literals_aux.withColumn("top_n", row_number().over(window_sub))

                    # Join df_emb_literals_aux with df_emb_literals to append new column with the number to top_n column
                    df_emb_literals = df_emb_literals.join(df_emb_literals_aux, (col('s') == col('s_')) & (col('count') == col('count_'))).drop('s_').drop('count_')

                # Filter by top N
                df_emb_literals = df_emb_literals.filter(col('top_n') <= self.top_n)

            df_emb_entities = self.spark_data.load_Embeddings_path(path[1])
            if self.top_n is not None:
                df_emb_entities = df_emb_entities.withColumn("count", lit(None).cast(StringType()))
                df_emb_entities = df_emb_entities.withColumn("top_n", lit(None).cast(StringType()))


            df_embeddings = df_emb_literals.union(df_emb_entities)


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
                    self.insert(tuples,table)
                    print(f'Inserted: {inserted}. Size {size}. Table: {table}')
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
        wait_time = 0
        while True:
            try:
                with psycopg2.connect(**config) as conn:
                    table_name = f'df_embs_{str.lower(self.config_experiment)}'
                    with conn.cursor() as cur:

                        # work_mem = '1GB'
                        # cur.execute('SET work_mem TO %s', (work_mem,))
                        #work_mem = '4GB'
                        #cur.execute('SET maintenance_work_mem TO %s', (work_mem,))
                        #max_parallel = '2'
                        #cur.execute('SET max_parallel_maintenance_workers to %s;', (max_parallel,))

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

                        # WHERE type_obj = 'L'
                        sql = f"""
                        SELECT df.sub, df.obj, df.type_obj, df.embedding, df.embedding <=> %s as distance
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
                                        # type_obj = 'L' AND
                                        sql = f"""
                                        SELECT sub, obj, type_obj, embedding, embedding <=> %s as distance 
                                        FROM {table_name}  
                                        WHERE 
                                        
                                        (embedding <=> %s) < 1.0
                                        ORDER BY distance LIMIT 10000;"""

                                        cur3.execute(sql, (vector, vector))
                                        rows_ = cur3.fetchall()#[:k]
                                        rows = list(set(sorted(rows_ + rows, key=lambda x: x[3]))) #[:k]


                        return rows #[(row[0], row[2]) for row in rows]

            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                print("Check the connection with database!")
                wait_time = wait_time + 60 if wait_time < 300 else wait_time
                print(f"Awaiting {wait_time} seconds!")
                time.sleep(wait_time)
