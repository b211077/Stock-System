package stock.model;

import java.sql.SQLException;
import java.util.ArrayList;

import stock.exception.NotExistException;
import stock.model.dto.StockDTO;

public class StockService {
	// 모든 주식 정보 반환
	public static ArrayList<StockDTO> getAllStock() throws SQLException {
		return StockDAO.getAllStock();
	}

	// 날짜로 주식 정보 검색
	public static StockDTO getStock(String name) throws SQLException, NotExistException {
		StockDTO stock = StockDAO.getStock(name);
		if (stock == null) {
			throw new NotExistException("검색하신 이름의 주식정보가 없습니다.");
		}
		return stock;
	}
}