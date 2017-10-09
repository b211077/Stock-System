DROP TABLE stock cascade constraint;

CREATE TABLE stock (
	cname 		VARCHAR2(30)  PRIMARY KEY,
	acc 			NUMBER(15) NOT NULL,
	decision 		VARCHAR2(10) NOT NULL
);