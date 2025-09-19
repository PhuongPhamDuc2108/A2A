import gradio as gr
import httpx
import asyncio
import os
import json
from dotenv import load_dotenv
import traceback

# Tải các biến môi trường và import HostAgent
load_dotenv()
from agent import HostAgent
# SỬA LỖI: Import các thành phần cần thiết từ a2a.utils
from a2a.types import AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


# Lớp quản lý trạng thái của các agent trong UI
class AgentManager:
    """Quản lý danh sách các remote agent được thêm vào từ UI."""

    def __init__(self):
        self.remote_agent_urls = set()

    # SỬA LỖI: Cập nhật hàm fetch_agent_card với logic bạn cung cấp
    async def fetch_agent_card(self, base_url: str) -> dict:
        """
        Lấy thông tin AgentCard từ endpoint .well-known một cách bất đồng bộ.
        """
        if not base_url.startswith(('http://', 'https://')):
            base_url = 'http://' + base_url.strip()

        # Đảm bảo URL không có dấu gạch chéo ở cuối trước khi nối đường dẫn
        well_known_url = f"{base_url.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(well_known_url)
                response.raise_for_status()
                # Trả về dữ liệu JSON để hàm add_agent_ui xử lý
                return response.json()
        except httpx.RequestError as e:
            return {"error": f"Không thể kết nối đến agent: {e}"}
        except Exception as e:
            return {"error": f"Lỗi không xác định: {e}"}

    def add_agent(self, url: str):
        """Thêm một URL agent mới vào danh sách."""
        if url:
            self.remote_agent_urls.add(url.strip())
        return list(self.remote_agent_urls)

    def get_agents_as_str(self):
        """Trả về danh sách URL agent dưới dạng chuỗi cho HostAgent."""
        return ",".join(self.remote_agent_urls)


# Khởi tạo đối tượng quản lý
agent_manager = AgentManager()


# --- Các hàm xử lý sự kiện cho Gradio ---

async def add_agent_ui(url: str):
    if not url:
        return list(agent_manager.remote_agent_urls), gr.update(value="URL không được để trống.")

    card_data = await agent_manager.fetch_agent_card(url)
    if "error" in card_data:
        return list(agent_manager.remote_agent_urls), json.dumps(card_data, indent=2)

    # Nếu không có lỗi, parse dữ liệu thành đối tượng AgentCard để xác thực
    try:
        AgentCard(**card_data)
        updated_list = agent_manager.add_agent(url)
        return updated_list, json.dumps(card_data, indent=2)
    except Exception as e:
        error_info = {"error": f"Dữ liệu Agent Card không hợp lệ: {e}"}
        return list(agent_manager.remote_agent_urls), json.dumps(error_info, indent=2)


async def user_ask(user_message: str, history: list):
    """
    Cập nhật UI ngay khi người dùng gửi tin nhắn.
    """
    return "", history + [[user_message, None]]


async def bot_respond(history: list):
    """
    Lấy câu trả lời từ HostAgent và stream vào giao diện.
    """
    user_message = history[-1][0]

    if not agent_manager.get_agents_as_str():
        history[-1][1] = "Vui lòng thêm ít nhất một remote agent trong tab 'Quản lý Agents' trước khi bắt đầu chat."
        yield history
        return

    os.environ["REMOTE_AGENT_URLS"] = agent_manager.get_agents_as_str()

    try:
        host_agent = await HostAgent.create()

        full_response = ""
        history[-1][1] = ""

        async for chunk in host_agent.stream(user_message, "gradio-session"):
            content_part = chunk.get('content', '')
            full_response += content_part
            history[-1][1] = full_response
            yield history

    except Exception as e:
        error_message = f"Đã xảy ra lỗi: {e}\n"
        error_message += traceback.format_exc()
        history[-1][1] = error_message
        yield history


# --- Xây dựng giao diện Gradio ---

with gr.Blocks(theme=gr.themes.Soft(), title="Host Agent UI") as demo:
    gr.Markdown("# Giao diện tương tác với Host Agent")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=600,
                bubble_full_width=False
            )
            msg_textbox = gr.Textbox(
                placeholder="Hỏi Host Agent điều gì đó và nhấn Enter...",
                label="Your Message",
                lines=2,
                scale=7
            )

            msg_textbox.submit(
                user_ask,
                [msg_textbox, chatbot],
                [msg_textbox, chatbot],
                queue=False,
            ).then(
                bot_respond, chatbot, chatbot
            )

        with gr.TabItem("Quản lý Agents"):
            gr.Markdown("## Thêm và xem các Remote Agent")
            with gr.Row():
                agent_url_input = gr.Textbox(label="URL của Remote Agent", placeholder="http://localhost:10001",
                                             scale=4)
                add_agent_btn = gr.Button("Thêm Agent", scale=1)

            gr.Markdown("### Danh sách Agent đã thêm")
            agent_list_display = gr.JSON(label="Agent URLs")

            gr.Markdown("### Thông tin Agent Card (JSON)")
            agent_card_json = gr.JSON(label="Agent Card Details")

            add_agent_btn.click(
                add_agent_ui,
                inputs=[agent_url_input],
                outputs=[agent_list_display, agent_card_json]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)