import unittest
from basic_footer import get_footer_content

class TestFooterContent(unittest.TestCase):
    def test_get_footer_content(self):
        expected_content = "&copy; 2023 My Website. All rights reserved."
        with open('footer_config.json', 'w') as file:
            json.dump({'footer_content': expected_content}, file)
        
        content = get_footer_content()
        self.assertEqual(content, expected_content)

if __name__ == '__main__':
    unittest.main()
