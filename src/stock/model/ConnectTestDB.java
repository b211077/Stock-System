package stock.model;
import java.sql.*;

public class ConnectTestDB {

	public static Connection getConnection(){
		Connection conn = null;
		
		try{
			Class.forName("com.mysql.jdbc.Driver");
			String jdbcUrl= "jdbc:mysql://dbstock.cxnc2jzecful.ap-northeast-2.rds.amazonaws.com:3306/innodb?autoReconnect=true";
			String userId = "stockManager";
			String userPass = "stockstock123";
			
			conn = DriverManager.getConnection(jdbcUrl,userId, userPass);
			System.out.println("Connected database successfully");
	/*		stmt = conn.createStatement();
		
			System.out.println("connect");
			
			String sql = "SELECT *from stock_info";
			ResultSet rs = stmt.executeQuery(sql);
			
			while(rs.next()){
				System.out.println(rs.getString(1)+rs.getInt(2));
			}
			
			rs.close();
			stmt.close();*/
			//conn.close();
			
		}catch(Exception e){
			System.out.println("SQL: Exception : " + e.getMessage());
		}
		return conn;		
	}

}
