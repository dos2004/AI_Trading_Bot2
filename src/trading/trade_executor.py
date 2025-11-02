"""
交易执行器
负责执行开仓、平仓等交易操作
"""
import time
from typing import Dict, Any, Optional
from src.api.binance_client import BinanceClient
from src.utils.decorators import retry_on_failure, log_execution


class TradeExecutor:
    """交易执行器"""
    
    def __init__(self, client: BinanceClient, config: Dict[str, Any]):
        """
        初始化交易执行器
        
        Args:
            client: Binance API客户端
            config: 交易配置
        """
        self.client = client
        self.config = config
        self.position_manager = None  # 将在外部设置
    
    # ==================== 开仓 ====================
    
    @log_execution
    @retry_on_failure(max_retries=3, delay=1)
    def open_long(self, symbol: str, quantity: float, leverage: int = None, 
                  take_profit: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        开多仓
        
        Args:
            symbol: 交易对
            quantity: 数量
            leverage: 杠杆倍数（1-100）
            take_profit: 止盈价
            stop_loss: 止损价
            
        Returns:
            订单信息
        """
        # 调整杠杆
        if leverage and leverage > 1:
            try:
                self.client.change_leverage(symbol, leverage)
                time.sleep(0.5)  # 等待杠杆调整生效
            except Exception as e:
                print(f"⚠️ 调整杠杆失败（继续开仓）: {e}")
        
        # 开仓
        try:
            order = self.client.create_market_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity
            )
            
            print(f"✅ 开多仓成功: {symbol} {quantity}")
            
            # 设置止盈止损
            if take_profit or stop_loss:
                time.sleep(1)  # 等待订单成交
                self._set_take_profit_stop_loss(symbol, 'BUY', quantity, 
                                                take_profit, stop_loss)
            
            return order
        except Exception as e:
            print(f"❌ 开多仓失败: {e}")
            raise
    
    @log_execution
    @retry_on_failure(max_retries=3, delay=1)
    def open_short(self, symbol: str, quantity: float, leverage: int = None,
                  take_profit: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        开空仓
        
        Args:
            symbol: 交易对
            quantity: 数量
            leverage: 杠杆倍数
            take_profit: 止盈价（价格下跌到这个价位止盈）
            stop_loss: 止损价（价格上涨到这个价位止损）
        """
        # 调整杠杆
        if leverage and leverage > 1:
            try:
                self.client.change_leverage(symbol, leverage)
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ 调整杠杆失败（继续开仓）: {e}")
        
        # 开仓
        try:
            order = self.client.create_market_order(
                symbol=symbol,
                side='SELL',
                quantity=quantity
            )
            
            print(f"✅ 开空仓成功: {symbol} {quantity}")
            
            # 设置止盈止损
            if take_profit or stop_loss:
                time.sleep(1)
                self._set_take_profit_stop_loss(symbol, 'SELL', quantity,
                                                take_profit, stop_loss)
            
            return order
        except Exception as e:
            print(f"❌ 开空仓失败: {e}")
            raise
    
    # ==================== 平仓 ====================
    
    @log_execution
    @retry_on_failure(max_retries=3, delay=1)
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        平仓（平掉整个持仓）
        
        会自动判断当前持仓方向并执行反向操作
        """
        try:
            # 获取当前持仓
            position = self.client.get_position(symbol)
            if not position or float(position['positionAmt']) == 0:
                print(f"⚠️ {symbol} 无持仓")
                return None
            
            # 确定平仓方向（与持仓相反）
            amount = abs(float(position['positionAmt']))
            side = 'SELL' if position['positionAmt'][0] != '-' else 'BUY'
            
            # 撤销所有挂单
            try:
                self.client.cancel_all_orders(symbol)
            except:
                pass
            
            # 平仓
            order = self.client.create_market_order(
                symbol=symbol,
                side=side,
                quantity=amount
            )
            
            print(f"✅ 平仓成功: {symbol} {side} {amount}")
            return order
            
        except Exception as e:
            print(f"❌ 平仓失败 {symbol}: {e}")
            raise
    
    def close_position_partial(self, symbol: str, percentage: float) -> Dict[str, Any]:
        """
        部分平仓
        
        Args:
            symbol: 交易对
            percentage: 平仓比例（0.1 = 10%）
        """
        if not 0 < percentage <= 1:
            raise ValueError("平仓比例必须在0-1之间")
        
        try:
            position = self.client.get_position(symbol)
            if not position or float(position['positionAmt']) == 0:
                print(f"⚠️ {symbol} 无持仓")
                return None
            
            total_amount = abs(float(position['positionAmt']))
            close_amount = total_amount * percentage
            
            # 确定平仓方向
            side = 'SELL' if position['positionAmt'][0] != '-' else 'BUY'
            
            order = self.client.create_market_order(
                symbol=symbol,
                side=side,
                quantity=close_amount
            )
            
            print(f"✅ 部分平仓成功: {symbol} {close_amount} ({percentage*100}%)")
            return order
            
        except Exception as e:
            print(f"❌ 部分平仓失败 {symbol}: {e}")
            raise
    
    def force_close_position(self, symbol: str, reason: str) -> Dict[str, Any]:
        """
        强制平仓（风控触发）
        
        Args:
            symbol: 交易对
            reason: 强制平仓原因
        """
        print(f"🚨 强制平仓: {symbol}, 原因: {reason}")
        return self.close_position(symbol)
    
    # ==================== 止盈止损 ====================
    
    def _set_take_profit_stop_loss(self, symbol: str, side: str, quantity: float,
                                   take_profit: float = None, stop_loss: float = None):
        """设置止盈止损"""
        try:
            orders = self.client.set_take_profit_stop_loss(
                symbol=symbol,
                side=side,
                quantity=quantity,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss
            )
            
            if take_profit:
                print(f"   📈 止盈价: ${take_profit:.2f}")
            if stop_loss:
                print(f"   🛑 止损价: ${stop_loss:.2f}")
                
        except Exception as e:
            print(f"⚠️ 设置止盈止损失败: {e}")
