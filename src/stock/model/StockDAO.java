package stock.model;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.ResourceBundle;

import stock.model.dto.StockDTO;
import stock.model.util.DBUtil;

public class StockDAO {
	static ResourceBundle sql = DBUtil.getResourceBundle();


	// �̸����� �ش� �ֽ��� ��� ���� ��ȯ
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
				stock = new StockDTO(rset.getString(1), rset.getInt(2), rset.getString(3));
			}
		} finally {
			DBUtil.close(con, pstmt, rset);
		}
		return stock;
	}

	// ��� �ֽ� �˻��ؼ� ��ȯ
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
				list.add(new StockDTO(rset.getString(1), rset.getInt(2), rset.getString(3)));
			}
		} finally {
			DBUtil.close(con, pstmt, rset);
		}
		return list;
	}
}
