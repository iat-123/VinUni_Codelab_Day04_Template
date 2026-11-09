"""
Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import unicodedata
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý tư vấn sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup.
Giao tiếp bằng tiếng Việt, thân thiện, chuyên nghiệp và chính xác.

## Công cụ khả dụng
- search_product_catalog(category, max_price): tra cứu danh mục xe điện hoặc du lịch.
- submit_support_ticket(customer_name, issue_description, priority): ghi nhận yêu cầu hỗ trợ.

## Quy tắc
- Không tự bịa giá, thông số, tình trạng sản phẩm hoặc mã ticket.
- Khi cần thông tin sản phẩm/giá, hãy dùng search_product_catalog và chỉ trình bày dữ liệu trả về.
- Chỉ xác nhận đã tiếp nhận hỗ trợ sau khi submit_support_ticket trả về kết quả.
- Nếu không có dữ liệu phù hợp, nói rõ điều đó và không suy đoán.
- Chỉ hỗ trợ nội dung liên quan đến Vingroup; lịch sự từ chối yêu cầu ngoài phạm vi.
- Không tiết lộ suy luận nội bộ. Chỉ cung cấp kết luận và thông tin cần thiết cho khách hàng.

## Định dạng trả lời
Theo dõi các bước bằng trace nội bộ (nhận yêu cầu, hành động/tool, kết quả);
phản hồi cuối cùng ngắn gọn, nêu kết quả và các thông tin quan trọng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text.lower())
        without_diacritics = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return without_diacritics.replace("đ", "d")

    @staticmethod
    def _extract_max_price(text: str):
        normalized = ToolCallingAgent._normalize(text)
        amount_pattern = (
            r"(\d+(?:[.,]\d+)*)\s*"
            r"(tram\s*trieu|triệu|trieu|tỷ|ty|đồng|vnd)?"
        )
        contextual_match = re.search(
            r"(?:gia|duoi|toi da|khong qua|ngan sach|tam gia)\D{0,30}"
            + amount_pattern,
            normalized
        )
        matches = list(re.finditer(amount_pattern, normalized))
        match = contextual_match
        if not match and matches:
            match = next(
                (candidate for candidate in reversed(matches) if candidate.group(2)),
                matches[-1]
            )
        if not match:
            return None

        number_text = match.group(1)
        unit = (match.group(2) or "").replace(" ", "")
        if "," in number_text or "." in number_text:
            parts = re.split(r"[,.]", number_text)
            if len(parts[-1]) <= 2:
                number = float(number_text.replace(",", "."))
            else:
                number = int("".join(parts))
        else:
            number = int(number_text)

        multiplier = {
            "tramtrieu": 100_000_000,
            "triệu": 1_000_000,
            "trieu": 1_000_000,
            "tỷ": 1_000_000_000,
            "ty": 1_000_000_000,
        }.get(unit, 1)
        return int(number * multiplier)

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        normalized = self._normalize(user_input)
        wants_catalog = any(word in normalized for word in (
            "xem", "tim", "mua", "gia", "bao nhieu", "duoi", "toi da",
            "tu van", "san pham", "resort", "khach san", "nghi duong",
            "du lich"
        ))
        is_electric_vehicle = any(word in normalized for word in (
            "xe dien", "vinfast", "o to dien", "xe vinfast"
        ))
        is_travel = any(word in normalized for word in (
            "du lich", "vinpearl", "resort", "khach san", "nghi duong"
        ))

        needs_catalog = wants_catalog and (is_electric_vehicle or is_travel)
        needs_ticket = any(word in normalized for word in (
            "can xu ly", "gap", "bi loi", "bi hong",
            "khieu nai", "su co", "gui yeu cau", "tao ticket"
        ))
        if "bao hanh" in normalized and any(word in normalized for word in (
            "loi", "hong", "kiem tra", "yeu cau"
        )):
            needs_ticket = True

        category = "xe_dien" if is_electric_vehicle else "du_lich"
        name_match = re.search(
            r"(?:tôi\s+tên|tên\s+tôi\s+là|toi\s+ten|ten\s+toi\s+la)\s+([^,.;]+)",
            user_input,
            re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        urgency_terms = ("nghiêm trọng", "gấp", "khẩn cấp", "urgent", "serious")
        priority = "high" if any(term in user_input.lower() for term in urgency_terms) else "medium"
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "category": category,
            "max_price": self._extract_max_price(user_input),
            "customer_name": customer_name,
            "issue_description": user_input.strip(),
            "priority": priority,
        }

    def _format_answer(
        self,
        observations: Dict[str, Any],
        intents: Dict[str, Any],
        user_input: str
    ) -> str:
        answers = []
        if "catalog" in observations:
            products = observations["catalog"]
            if not products or (len(products) == 1 and "error" in products[0]):
                answers.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
            else:
                descriptions = [
                    f"{product['name']} — {product['price_vnd']:,} VNĐ"
                    for product in products
                ]
                answers.append("Sản phẩm phù hợp: " + "; ".join(descriptions) + ".")

        if "ticket" in observations:
            ticket = observations["ticket"]
            answers.append(
                f"Đã tạo yêu cầu hỗ trợ cho {ticket['customer_name']}. "
                f"Mã ticket: {ticket['ticket_id']} (ưu tiên {ticket['priority']}, "
                f"trạng thái {ticket['status']})."
            )

        if not answers:
            normalized = self._normalize(user_input)
            if "bao hanh" in normalized and "pin" in normalized:
                answers.append(
                    "Theo dữ liệu catalog hiện có, VinFast VF 5 Plus ghi nhận "
                    "bảo hành pin 10 năm. Chính sách cụ thể có thể tùy mẫu xe "
                    "và điều kiện áp dụng; vui lòng xác nhận với VinFast."
                )
            else:
                answers.append(
                    "Tôi có thể hỗ trợ tra cứu xe điện, dịch vụ du lịch hoặc "
                    "ghi nhận yêu cầu hỗ trợ của Vingroup. Bạn cần trợ giúp nội dung nào?"
                )
        return " ".join(answers)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        self.trace.append({
            "step": "intent_detection",
            "user_input": user_input,
            "intents": intents
        })

        actions = []
        if intents["needs_catalog"]:
            actions.append("catalog")
        if intents["needs_ticket"]:
            actions.append("ticket")
        if not actions:
            actions.append("faq")

        observations = {}
        if self.max_iterations <= 0:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        for iteration, action in enumerate(actions, start=1):
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": self.max_iterations,
                    "status": "max_iterations_reached"
                }

            if action == "catalog":
                arguments = {"category": intents["category"]}
                if intents["max_price"] is not None:
                    arguments["max_price"] = intents["max_price"]
                result = TOOL_MAP["search_product_catalog"](**arguments)
                observations["catalog"] = result
                self.trace.append({
                    "step": iteration,
                    "action": "search_product_catalog",
                    "arguments": arguments,
                    "observation": result
                })
            elif action == "ticket":
                arguments = {
                    "customer_name": intents["customer_name"],
                    "issue_description": intents["issue_description"],
                    "priority": intents["priority"]
                }
                result = TOOL_MAP["submit_support_ticket"](**arguments)
                observations["ticket"] = result
                self.trace.append({
                    "step": iteration,
                    "action": "submit_support_ticket",
                    "arguments": arguments,
                    "observation": result
                })
            else:
                self.trace.append({
                    "step": iteration,
                    "action": "answer_faq",
                    "observation": "No tool required."
                })

        answer = self._format_answer(observations, intents, user_input)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": len(actions),
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
