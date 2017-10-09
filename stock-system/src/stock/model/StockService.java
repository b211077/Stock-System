package stock.model;

import java.sql.SQLException;
import java.util.ArrayList;

import stock.exception.NotExistException;
import stock.model.dto.StockDTO;

public class StockService {
	// ��� �ֽ� ���� ��ȯ
	public static ArrayList<StockDTO> getAllStock() throws SQLException {
		return StockDAO.getAllStock();
	}

	// ��¥�� �ֽ� ���� �˻�
	public static StockDTO getStock(String name) throws SQLException, NotExistException {
		StockDTO stock = StockDAO.getStock(name);
		if (stock == null) {
			throw new NotExistException("�˻��Ͻ� �̸��� �ֽ������� �����ϴ�.");
		}
		return stock;
	}
}