import pandas
import numpy as np
import mysql.connector as dbconnector
from collections import OrderedDict


class maria(object):
	# __instance = None
	__host = None
	__user = None
	__password = None
	__database = None
	__port = None
	__session = None
	__connection = None

	# def __new__(cls, *args, **kwargs):
	#   if not cls.__instance or not cls.__database:
	#        cls.__instance = super(maria, cls).__new__(cls, *args, **kwargs)
	#    return cls.__instance
	# End def __new__

	def __init__(self, host="192.168.233.134", user="root", password="Infor3233!@#", database="innostock", port=''):
		self.__instance = self
		self.__host = host
		self.__user = user
		self.__password = password
		self.__database = database
		self.__port = port

	## End def __init__

	def __open(self):
		try:
			cnx = dbconnector.connect(host=self.__host, user=self.__user, password=self.__password,
			                          database=self.__database)
			self.__connection = cnx
			self.__session = cnx.cursor()
		except dbconnector.Error as e:
			print("Error %d: %s" % (e.args[0], e.args[1]))

	## End def __open

	def __close(self):
		self.__session.close()
		self.__connection.close()

	## End def __close


	def select(self, strquery, values=[]):
		# print(strquery, ' : ', values)
		if not strquery:
			print('Query is empty.')
			return -1

		self.__open()

		try:
			self.__session.execute(strquery, values)

		except dbconnector.Error as e:
			print("Error %d: %s" % (e.args[0], e.args[1]))
			return -1

		result = pandas.DataFrame(self.__session.fetchall(), columns=[x[0] for x in self.__session.description])

		self.__close()

		return result

	## End def select

	def update(self, table, df_source, key_colname, single_commit=False):
		if not len(df_source):
			print("DataFrame 'df_source' is empty.")
			return -1

		self.__open()

		values = None
		df_source.fillna('')
		df_source = df_source.set_index(key_colname)
		for ix, r in df_source.iterrows():
			source = r.to_dict()
			keys = source.keys()
			values = list(source.values()) + [ix]

			query = "UPDATE %s SET " % table
			l = len(keys) - 1
			for i, key in enumerate(keys):
				query += key + " = %s" + (', ' if i < l else ' ')
			## End for keys
			query += 'WHERE ' + key_colname + ' = %s'

			# print(query, ', ', values)

			try:
				self.__session.execute(query, values)

			except dbconnector.Error as e:
				self.__connection.rollback()
				print("Error %d: %s" % (e.args[0], e.args[1]))
				return -1

			if single_commit:
				self.__connection.commit()

		## End for loop

		if not single_commit:
			self.__connection.commit()

		self.__close()

		return self.__session.lastrowid

	## End def update

	def update_single(self, table, where=None, *args, **kwargs):
		query = "UPDATE %s SET " % table
		keys = kwargs.keys()
		values = tuple(kwargs.values()) + tuple(args)
		l = len(keys) - 1
		for i, key in enumerate(keys):
			query += "`" + key + "` = %s"
			if i < l:
				query += ","
			## End if i less than 1
		## End for keys
		query += " WHERE %s" % where

		self.__open()
		self.__session.execute(query, values)
		self.__connection.commit()

		# Obtain rows affected
		update_rows = self.__session.rowcount
		self.__close()

		return update_rows

	## End function update


	def insert(self, table, df_source, single_commit=False):
		if not len(df_source):
			print("DataFrame 'df_source' is empty.")
			return -1

		self.__open()

		values = None
		for ix, r in df_source.iterrows():
			source = r.to_dict()
			keys = source.keys()
			values = tuple(source.values())

			query = "INSERT INTO %s " % table
			query += "(" + ",".join(["%s"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(
				["%s"] * len(values)) + ")"

			# print(ix, "query : ", query, "   values : ", values)

			try:
				self.__session.execute(query, values)

			except dbconnector.Error as e:
				self.__connection.rollback()
				print("Error %d: %s" % (e.args[0], e.args[1]))
				return -1

			if single_commit:
				self.__connection.commit()

		## End for loop

		if not single_commit:
			self.__connection.commit()

		self.__close()

		return self.__session.lastrowid

	## End def update

	def execute(self, query, values=[], commit=True):
		self.__open()

		try:
			self.__session.execute(query, values)

		except dbconnector.Error as e:
			if commit:
				self.__connection.rollback()
				self.__close()

			print("Error %d: %s" % (e.args[0], e.args[1]))

			return -1

		if commit:
			self.__connection.commit()

		self.__close()

		return 1

	## End function update


	def callproc(self, procname, arg):
		self.__open()
		try:
			self.__session.callproc(procname, arg)

		except dbconnector.Error as e:
			self.__close()
			print("Error %d: %s" % (e.args[0], e.args[1]))
			return -1

		self.__close()

		return 1
		## End function callproc

		## End class