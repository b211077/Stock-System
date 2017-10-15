package stock.model;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.ResourceBundle;

import stock.model.dto.StockDTO;
import stock.model.util.DBUtil;
import stock.model.dto.*;
public class StockDAO {
	static ResourceBundle sql = DBUtil.getResourceBundle();

	public static ArrayList<StockDTO> getSinfo(){
		
		ArrayList<StockDTO> search_list = new ArrayList<StockDTO>();
		Connection conn = null;
		PreparedStatement pstm = null;
		ResultSet rs = null;
		int data_num = 4; //712변경예정
		
		try {
			conn = ConnectTestDB.getConnection();
			System.out.println("2");
			pstm = conn.prepareStatement("select * from stock_info");
			System.out.println("11");
			rs = pstm.executeQuery();
			
			for(int i=0; i<data_num; i++){
				rs.next();
				search_list.add(new StockDTO(rs.getString(1),rs.getInt(2),rs.getString(3),rs.getInt(4)));
				System.out.println("1");
			}
			System.out.println(search_list);
		
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return search_list;
	}
	
	public static StockDTO getRinfo(String i){
		
		StockDTO search_result = new StockDTO();
		Connection conn = null;
		PreparedStatement pstm = null;
		ResultSet rs = null;
		
		
		try {
			conn = ConnectTestDB.getConnection();
			pstm = conn.prepareStatement("select * from stock_info");
			rs = pstm.executeQuery();

				rs.next();
				search_result=new StockDTO(rs.getString(1),rs.getInt(2),rs.getString(3),rs.getInt(4));

		
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return search_result;
	}
	// 이름으로 해당 주식의 모든 정보 반환
	public static StockDTO getStock(String name) throws SQLException {
		Connection con = null;
		PreparedStatement pstmt = null;
		ResultSet rset = null;
		StockDTO stock = null;
		try {
			con = DBUtil.getConnection();
			pstmt = con.prepareStatement(sql.getString("getStock"));
			pstmt.setString(1, name);
			rset = pstmt.executeQuery();
			if (rset.next()) {
				stock = new StockDTO(rset.getString(1), rset.getInt(2), rset.getString(3), rset.getInt(4));
			}
		} finally {
			DBUtil.close(con, pstmt, rset);
		}
		return stock;
	}

	// 모든 주식 검색해서 반환
	public static ArrayList<StockDTO> getAllStock() throws SQLException {
		Connection con = null;
		PreparedStatement pstmt = null;
		ResultSet rset = null;
		ArrayList<StockDTO> list = null;
		try {
			con = DBUtil.getConnection();
			pstmt = con.prepareStatement(sql.getString("getAllStock"));
			rset = pstmt.executeQuery();

			list = new ArrayList<StockDTO>();
			while (rset.next()) {
				list.add(new StockDTO(rset.getString(1), rset.getInt(2), rset.getString(3), rset.getInt(4)));
			}
		} finally {
			DBUtil.close(con, pstmt, rset);
		}
		return list;
	}
}
